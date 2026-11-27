import asyncio
import importlib.util
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "aws_live_test", Path(__file__).parents[1] / "scripts/aws_live_test.py"
)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def test_unpriced_or_ambiguous_compute_rate_blocks_launch():
    prices = MagicMock()
    prices.get_paginator.return_value.paginate.return_value = [{"PriceList": []}]
    with pytest.raises(ValueError, match="one positive"):
        live.price(prices, "Hrs", {})


@pytest.mark.parametrize(("profile", "source"), [("default", "sso"), (None, "env")])
def test_plan_selects_the_requested_credential_provider(tmp_path, monkeypatch, profile, source):
    pytest.importorskip("boto3")
    from botocore.httpsession import URLLib3Session

    config = tmp_path / "config"
    config.write_text(
        "[default]\nsso_session = test\nsso_account_id = 111111111111\nsso_role_name = TokenFlowOperator\n"
        "[sso-session test]\nsso_start_url = https://example.invalid/start\nsso_region = us-east-1\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "unused-credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "offline")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "offline")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.setattr(URLLib3Session, "send", lambda *_: pytest.fail("Unexpected AWS connection"))

    def inspect_session(session, args):
        assert session.region_name == "us-east-1"
        assert session.get_credentials().method == source
        return {}

    monkeypatch.setattr(live, "preflight", inspect_session)
    asyncio.run(live.run(SimpleNamespace(profile=profile, execute=False)))


def test_estimate_rounds_up_and_no_overbudget_launch(monkeypatch):
    session = MagicMock()
    session.get_credentials.return_value.method = "sso"
    session.client.return_value.describe_images.return_value = {"Images": [
        {"State": "available", "BlockDeviceMappings": [{"Ebs": {}}]},
    ]}
    monkeypatch.setattr(live, "verify_quota", lambda *_: {"quota_vcpus": 4, "committed_vcpus": 0})
    monkeypatch.setattr(live, "promotion_balance", lambda *_: Decimal(50))
    monkeypatch.setattr(live, "check_network", lambda *_: None)
    monkeypatch.setattr(live, "volume_size", lambda *_: 60)
    monkeypatch.setattr(live, "price", lambda _, unit, __: Decimal("0.526") if unit == "Hrs" else Decimal("0.08"))
    args = SimpleNamespace(execute=False, expected_account=None, credit_description=None, ami="ami-test")
    report = live.preflight(session, args)
    assert report["estimated_usd"] == "3.07" and report["credential_source"] == "sso"
    monkeypatch.setattr(live, "price", lambda *_: Decimal(20))
    with pytest.raises(ValueError, match="exceeds"):
        live.preflight(session, args)


@pytest.mark.parametrize("expected_account", [None, "222222222222"])
def test_execute_checks_the_approved_account_before_reading_billing_or_ec2(expected_account):
    session = MagicMock()
    session.client.return_value.get_caller_identity.return_value = {"Account": "111111111111"}
    with pytest.raises(ValueError, match="expected-account"):
        live.preflight(session, SimpleNamespace(execute=True, expected_account=expected_account))
    if expected_account:
        session.client.assert_called_once_with("sts")
    else:
        session.client.assert_not_called()


def test_live_credit_excludes_another_account_and_expired_awards():
    billing = MagicMock()
    now = live.datetime.now(live.UTC)
    award = {
        "accountId": "123", "creditType": "Promotion", "creditStatus": "ENABLED",
        "startDate": now - live.timedelta(days=10), "endDate": now + live.timedelta(days=30),
        "applicableProductNames": ["Amazon Elastic Compute Cloud"], "description": "Compute award",
        "remainingAmount": {"currencyCode": "USD", "currencyAmount": "85"},
        "estimatedAmount": {"currencyCode": "USD", "currencyAmount": "67"},
    }
    billing.get_credits.return_value = {"credits": [
        award, {**award, "endDate": now}, {**award, "accountId": "456"},
        {**award, "exhaustDate": now + live.timedelta(minutes=30)},
        {**award, "applicableProductNames": []}, {**award, "creditStatus": "DISABLED"},
    ]}
    assert live.promotion_balance(billing, "123", "Compute award") == 67


@pytest.mark.parametrize("change", [
    {"DeleteOnTermination": False}, {"VolumeType": "io2"}, {"Iops": 6000}, {"Throughput": 300},
])
def test_undeletable_or_unpriced_volume_is_rejected(change):
    disk = {"DeleteOnTermination": True, "VolumeType": "gp3", "VolumeSize": 100, **change}
    image = {"Architecture": "x86_64", "RootDeviceType": "ebs", "PlatformDetails": "Linux/UNIX",
             "BlockDeviceMappings": [{"Ebs": disk}]}
    with pytest.raises(ValueError, match="baseline gp3"):
        live.volume_size(image)


def test_missing_guard_and_replacement_launch_fail_closed():
    sfn = MagicMock()
    sfn.describe_execution.return_value = {"status": "FAILED"}
    with pytest.raises(RuntimeError, match="did not reach"):
        live.arm_guard(sfn, "machine", "role", "run")
    with pytest.raises(RuntimeError, match="one EC2 launch"):
        asyncio.run(live.AdoptOnly("run").launch(None, "new-key"))


def test_guard_verifies_role_can_terminate_before_gpu_launch():
    cfn, sfn, iam = MagicMock(), MagicMock(), MagicMock()
    cfn.get_template.return_value = {"TemplateBody": live.GUARD_TEMPLATE.read_text()}
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "CREATE_COMPLETE",
                                                    "Outputs": [{"OutputKey": "Machine", "OutputValue": "machine"}]}]}
    sfn.describe_state_machine.return_value = {"type": "STANDARD", "status": "ACTIVE", "roleArn": "role",
                                               "definition": json.dumps(live.guard_definition())}
    iam.simulate_principal_policy.return_value = {"EvaluationResults": [{"EvalDecision": "implicitDeny"}]}
    with pytest.raises(ValueError, match="permission"):
        live.verify_guard(cfn, sfn, iam, "stack", "tf-test-0123456789abcdef", "123456789012")


def test_private_vpn_ingress_allows_a_preloaded_worker_without_nat():
    ec2 = MagicMock()
    ec2.describe_subnets.return_value = {"Subnets": [{"VpcId": "vpc", "AvailabilityZone": "zone"}]}
    ec2.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc", "InstanceTenancy": "default"}]}
    ec2.describe_instance_type_offerings.return_value = {"InstanceTypeOfferings": [{}]}
    rule = {"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8001, "IpRanges": [{"CidrIp": "10.0.7.9/32"}]}
    ec2.describe_security_groups.return_value = {"SecurityGroups": [{"VpcId": "vpc", "IpPermissions": [rule]}]}
    args = SimpleNamespace(subnet="subnet", worker_group="worker", source_group=None, source_cidr="10.0.7.9/32")
    live.check_network(ec2, args)
    ec2.describe_vpcs.assert_called_once_with(VpcIds=["vpc"])
    ec2.describe_nat_gateways.assert_not_called()
    rule["IpRanges"].append({"CidrIp": "0.0.0.0/0"})
    with pytest.raises(ValueError, match="ingress"):
        live.check_network(ec2, args)


@pytest.mark.parametrize("vpcs", [
    {}, {"Vpcs": []}, {"Vpcs": [{"VpcId": "vpc"}]},
    {"Vpcs": [{"VpcId": "vpc", "InstanceTenancy": ""}]},
    {"Vpcs": [{"VpcId": "vpc", "InstanceTenancy": "dedicated"}]},
    {"Vpcs": [{"VpcId": "other", "InstanceTenancy": "default"}]},
])
def test_unverified_or_dedicated_vpc_fails_shared_tenancy_check(vpcs):
    ec2 = MagicMock()
    ec2.describe_subnets.return_value = {"Subnets": [{"VpcId": "vpc"}]}
    ec2.describe_vpcs.return_value = vpcs
    with pytest.raises(ValueError, match="verified default tenancy"):
        live.check_network(ec2, SimpleNamespace(subnet="subnet"))
    ec2.describe_instance_type_offerings.assert_not_called()


@pytest.fixture
def aws_run(monkeypatch, tmp_path):
    """Fake the service boundary. Keep provisioning, adoption, and ASGI routing real."""
    services = {name: MagicMock(name=name) for name in (
        "ec2", "cloudformation", "stepfunctions", "iam",
    )}
    session = MagicMock()
    session.client.side_effect = lambda name, **kwargs: services[name]
    sdk = MagicMock()
    sdk.Session.return_value = session
    monkeypatch.setitem(sys.modules, "boto3", sdk)
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=MagicMock()))
    # The separate price/credit/quota tests cover the read-only preflight.
    monkeypatch.setattr(live, "preflight", lambda *_: {
        "principal_is_root": False, "estimated_usd": "3.07", "account": "123456789012", "ebs_volumes": 1,
    })
    cfn, sfn, ec2 = (services[name] for name in ("cloudformation", "stepfunctions", "ec2"))
    cfn.get_template.return_value = {"TemplateBody": live.GUARD_TEMPLATE.read_text()}
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "CREATE_COMPLETE",
        "Outputs": [{"OutputKey": "Machine", "OutputValue": "machine"}]}]}
    sfn.describe_state_machine.return_value = {"type": "STANDARD", "status": "ACTIVE", "roleArn": "role",
                                               "definition": json.dumps(live.guard_definition())}
    sfn.describe_state_machine_for_execution.return_value = {
        "stateMachineArn": "machine", "roleArn": "role", "definition": json.dumps(live.guard_definition())}
    services["iam"].simulate_principal_policy.return_value = {"EvaluationResults": [{"EvalDecision": "allowed"}]}
    sfn.start_execution.return_value = {"executionArn": "execution"}
    sfn.describe_execution.return_value = {"status": "RUNNING"}
    sfn.get_execution_history.return_value = {"events": [{"type": "WaitStateEntered",
        "stateEnteredEventDetails": {"name": "WaitForDeadline"}}]}
    vm = {"InstanceId": "i-test", "State": {"Name": "running"}, "PrivateIpAddress": "10.5.6.7",
          "BlockDeviceMappings": [{"Ebs": {"VolumeId": "vol-test"}}],
          "NetworkInterfaces": [{"NetworkInterfaceId": "eni-test"}]}
    instances = {"Reservations": [{"Instances": [vm]}]}

    def launch(**parameters):
        sfn.get_execution_history.assert_called_once()
        assert parameters["MinCount"] == parameters["MaxCount"] == 1
        assert parameters["InstanceType"] == live.INSTANCE_TYPE
        assert parameters["NetworkInterfaces"][0]["AssociatePublicIpAddress"] is False
        assert parameters["NetworkInterfaces"][0]["DeleteOnTermination"] is True
        vm["Tags"] = parameters["TagSpecifications"][0]["Tags"]
        tags = {tag["Key"]: tag["Value"] for tag in vm["Tags"]}
        assert tags["TokenFlowNamespace"] == sfn.start_execution.call_args.kwargs["name"]
        return {"Instances": [vm]}

    ec2.run_instances.side_effect = launch
    ec2.get_paginator.return_value.paginate.return_value = [instances]
    ec2.describe_instances.return_value = instances

    def terminate(InstanceIds):
        assert InstanceIds == [vm["InstanceId"]]
        vm["State"]["Name"] = "terminated"

    def confirm_terminated(InstanceIds, WaiterConfig):
        assert InstanceIds == [vm["InstanceId"]]
        assert vm["State"]["Name"] == "terminated"

    ec2.terminate_instances.side_effect = terminate
    ec2.get_waiter.return_value.wait.side_effect = confirm_terminated
    ec2.describe_volumes.return_value = {"Volumes": []}
    ec2.describe_network_interfaces.return_value = {"NetworkInterfaces": []}
    args = SimpleNamespace(profile=None, ami="ami-abcdef", subnet="subnet-abc", worker_group="sg-abc",
        execute=True, confirm_private_route=True, local_checkpoint="/models/qwen", image=None,
        output=tmp_path, guard_stack="test-guard", namespace="tf-test-0123456789abcdef")
    return args, vm, ec2, sfn


@pytest.mark.parametrize("stage,field", [
    ("machine", "definition"), ("execution", "definition"),
    ("execution", "roleArn"), ("execution", "stateMachineArn"),
])
def test_changed_cleanup_workflow_blocks_gpu_launch(aws_run, stage, field):
    args, _, ec2, sfn = aws_run
    target = sfn.describe_state_machine if stage == "machine" else sfn.describe_state_machine_for_execution
    if field == "definition":
        # The execution would enter the expected wait and then finish without
        # terminating the instance. The stored CloudFormation template still matches.
        definition = json.loads(target.return_value[field])
        definition["States"]["WaitForDeadline"]["Next"] = "Finished"
        target.return_value[field] = json.dumps(definition)
    else:
        target.return_value[field] = "different-resource"
    with pytest.raises(ValueError, match="differs"):
        asyncio.run(live.run(args))
    ec2.run_instances.assert_not_called()
    if stage == "machine":
        sfn.start_execution.assert_not_called()
    else:
        sfn.stop_execution.assert_called_once_with(executionArn="execution", cause="Pre-launch guard verification failed")


def test_competing_invocations_cannot_share_an_armed_standard_execution(aws_run):
    args, _, ec2, sfn = aws_run
    attempts = []

    def start_execution(stateMachineArn, name, input):
        assert name == args.namespace
        # Standard workflows join a running execution for identical name and
        # input. A different input under the same name is rejected by AWS.
        attempt = json.loads(input)["attempt"]
        if attempts and attempt != attempts[0]:
            raise RuntimeError("ExecutionAlreadyExists")
        attempts.append(attempt)
        return {"executionArn": "execution"}

    sfn.start_execution.side_effect = start_execution
    live.arm_guard(sfn, "machine", "role", args.namespace)
    with pytest.raises(RuntimeError, match="ExecutionAlreadyExists"):
        asyncio.run(live.run(args))
    ec2.run_instances.assert_not_called()
    sfn.stop_execution.assert_not_called()


@pytest.mark.parametrize("incomplete_stream", [False, True])
def test_live_run_adopts_one_private_gpu_and_cleans_up_after_inference(aws_run, monkeypatch, incomplete_stream):
    args, vm, ec2, sfn = aws_run
    requests = []

    def engine(request):
        assert request.url.host == vm["PrivateIpAddress"]
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen-test"}]})
        if request.url.path == "/metrics":
            if request.url.port == 8001:
                return httpx.Response(200, text='DCGM_FI_DEV_GPU_UTIL{UUID="GPU-test"} 64')
            return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                                              "vllm:kv_cache_usage_perc 0.14")
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen-test"
        requests.append(payload.get("stream", False))
        if payload.get("stream"):
            content = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            if not incomplete_stream:
                content += b"data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    create_app = live.create_app
    monkeypatch.setattr(live, "create_app", lambda settings, **kwargs: create_app(
        settings.model_copy(update={"worker_health_interval_seconds": 0.05}),
        worker_transport=httpx.MockTransport(engine), **kwargs,
    ))
    if incomplete_stream:
        with pytest.raises(RuntimeError, match="Incomplete streaming"):
            asyncio.run(live.run(args))
    else:
        asyncio.run(live.run(args))
    assert requests == [False, True]
    ec2.run_instances.assert_called_once()
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-test"])
    ec2.get_waiter.return_value.wait.assert_called_once()
    ec2.describe_volumes.assert_called_once_with(Filters=[{"Name": "volume-id", "Values": ["vol-test"]}])
    ec2.describe_network_interfaces.assert_called_once_with(
        Filters=[{"Name": "network-interface-id", "Values": ["eni-test"]}])
    sfn.stop_execution.assert_called_once()
    out, = args.output.iterdir()
    receipt = json.loads((out / "run.json").read_text())
    assert receipt["guard"] == "execution"
    assert receipt["instance_id"] == "i-test"
    cleaned = json.loads((out / "cleanup.json").read_text())
    assert cleaned["instance_id"] == "i-test" and cleaned["volume_ids"] == ["vol-test"]
    assert cleaned["interface_ids"] == ["eni-test"]
    if incomplete_stream:
        assert not (out / "responses.json").exists()
    else:
        result = json.loads((out / "responses.json").read_text())
        assert result["chat"]["choices"][0]["message"]["content"] == "hello"
        assert result["stream"][-1] == "[DONE]"
        assert result["worker"]["telemetry"]["gpu_utilization"] == 0.64


def test_live_run_leaves_guard_armed_when_release_cannot_be_verified(aws_run, monkeypatch):
    args, _, ec2, sfn = aws_run

    async def interrupted(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(live, "check_gateway", interrupted)
    ec2.get_waiter.return_value.wait.side_effect = TimeoutError("EC2 termination unconfirmed")
    with pytest.raises(TimeoutError, match="termination unconfirmed"):
        asyncio.run(live.run(args))
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-test"])
    sfn.stop_execution.assert_not_called()


@pytest.mark.parametrize("becomes_visible", [True, False])
def test_lost_launch_reply_recovers_only_this_attempt_and_preserves_guard(aws_run, monkeypatch, becomes_visible):
    args, vm, ec2, sfn = aws_run
    accept_launch = ec2.run_instances.side_effect
    polls = []

    def lost_reply(**params):
        accept_launch(**params)
        raise TimeoutError("RunInstances lost its reply")

    def inventory(**kwargs):
        # Another key in the same namespace must never be terminated by this client.
        other = {**vm, "InstanceId": "i-other", "Tags": [
            {**tag, "Value": "f" * 32} if tag["Key"] == "TokenFlowKey" else tag for tag in vm["Tags"]
        ]}
        polls.append(kwargs["Filters"])
        rows = [other, vm] if becomes_visible and len(polls) > 1 else [other]
        return [{"Reservations": [{"Instances": rows}]}]

    async def skip_delay(seconds):
        pass

    monkeypatch.setattr(live.asyncio, "sleep", skip_delay)
    ec2.run_instances.side_effect = lost_reply
    ec2.get_paginator.return_value.paginate.side_effect = inventory
    with pytest.raises(TimeoutError, match="lost its reply"):
        asyncio.run(live.run(args))
    sfn.get_execution_history.assert_called_once()
    ec2.run_instances.assert_called_once()
    sfn.stop_execution.assert_not_called()
    out = args.output / args.namespace
    receipt = json.loads((out / "run.json").read_text())
    assert receipt["launch_key"] == next(tag["Value"] for tag in vm["Tags"] if tag["Key"] == "TokenFlowKey")
    if becomes_visible:
        ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-test"])
        ec2.get_waiter.return_value.wait.assert_called_once()
        assert json.loads((out / "cleanup.json").read_text())["instance_id"] == "i-test"
    else:
        ec2.terminate_instances.assert_not_called()
        assert not (out / "cleanup.json").exists()


@pytest.mark.parametrize("missing", ["BlockDeviceMappings", "NetworkInterfaces"])
def test_missing_attachment_inventory_keeps_guard_armed(aws_run, monkeypatch, missing):
    args, vm, ec2, sfn = aws_run
    vm[missing] = []

    async def completed(*args):
        return {"smoke": "passed"}

    monkeypatch.setattr(live, "check_gateway", completed)
    with pytest.raises(RuntimeError, match="attachment inventory"):
        asyncio.run(live.run(args))
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-test"])
    ec2.get_waiter.return_value.wait.assert_called_once()
    sfn.stop_execution.assert_not_called()
