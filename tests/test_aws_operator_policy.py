import json
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tokenflow.provisioner.base import PoolSpec
from tokenflow.provisioner.cloud import CloudProvisioner


SCRIPT = Path(__file__).parents[1] / "scripts/aws_operator_policy.py"
ACCOUNT = "0" * 12
REGION = "us-east-1"
WORKERS = [
    "--namespace", "tokenflow-demo", "--pool", "qwen",
    "--ami", "ami-12345678", "--subnet", "subnet-12345678",
    "--security-group", "sg-12345678", "--instance-type", "g4dn.xlarge",
]


def render(mode, *extras):
    return subprocess.run(
        [sys.executable, str(SCRIPT), mode, "--account-id", ACCOUNT, "--region", REGION, *extras],
        text=True, capture_output=True,
    )


def by_id(result):
    assert result.returncode == 0, result.stderr
    return {s["Sid"]: s for s in json.loads(result.stdout)["Statement"]}


def test_bootstrap_is_read_only_and_quota_writes_are_opt_in():
    bootstrap = by_id(render("bootstrap"))
    quota = bootstrap["RegionalGpuQuota"]
    assert quota["Resource"] == f"arn:aws:servicequotas:{REGION}:{ACCOUNT}:ec2/L-DB2E81BA"
    assert "servicequotas:RequestServiceQuotaIncrease" not in quota["Action"]
    assert set(quota["Action"]) == {"servicequotas:GetServiceQuota", "servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota"}
    assert bootstrap["ReadCreditsAndPrices"]["Action"] == ["billing:GetCredits", "pricing:GetProducts"]
    for statement in bootstrap.values():
        actions = statement["Action"]
        for action in [actions] if isinstance(actions, str) else actions:
            assert action.split(":")[1].startswith(("Describe", "Get", "List"))
    opted_in = by_id(render("bootstrap", "--allow-quota-request"))
    assert opted_in["RegionalGpuQuota"]["Resource"] == quota["Resource"]
    assert set(opted_in["RegionalGpuQuota"]["Action"]) - set(quota["Action"]) == {"servicequotas:RequestServiceQuotaIncrease"}


def test_worker_permissions_match_tagged_private_launch_and_scoped_termination(tmp_path):
    output = tmp_path / "operator.json"
    assert render("workers", *WORKERS, "--output", str(output)).returncode == 0
    statements = {s["Sid"]: s for s in json.loads(output.read_text(encoding="utf-8"))["Statement"]}
    ec2 = f"arn:aws:ec2:{REGION}:{ACCOUNT}"
    assert set(statements["UseApprovedWorkerInputs"]["Resource"]) == {
        f"arn:aws:ec2:{REGION}::image/ami-12345678", f"{ec2}:subnet/subnet-12345678", f"{ec2}:security-group/sg-12345678",
    }
    assert statements["CreatePrivateWorkerInterface"]["Condition"] == {
        "ArnEquals": {"ec2:Subnet": f"{ec2}:subnet/subnet-12345678"},
        "Bool": {"ec2:AssociatePublicIpAddress": "false"},
    }
    launch = statements["LaunchNamedWorker"]
    assert launch["Resource"] == f"{ec2}:instance/*"
    assert launch["Condition"]["StringEquals"]["ec2:InstanceType"] == "g4dn.xlarge"
    assert launch["Condition"]["StringEquals"]["aws:RequestTag/TokenFlowNamespace"] == "tokenflow-demo"
    assert launch["Condition"]["StringEquals"]["aws:RequestTag/TokenFlowPool"] == "qwen"
    assert statements["ApplyWorkerTagsOnlyAtLaunch"]["Condition"]["StringEquals"]["ec2:CreateAction"] == "RunInstances"
    assert statements["TerminateOwnedWorkers"]["Condition"]["StringEquals"] == {
        "ec2:ResourceTag/TokenFlowNamespace": "tokenflow-demo", "ec2:ResourceTag/TokenFlowPool": "qwen",
    }
    assert '"iam:' not in output.read_text()
    for statement in statements.values():
        if statement["Action"] in ("ec2:RunInstances", "ec2:CreateTags", "ec2:TerminateInstances"):
            assert statement["Resource"] != "*"


def test_policy_covers_actual_provider_launch_without_standalone_tagging(monkeypatch):
    """Capture the production RunInstances call at the SDK boundary, without AWS."""
    flags = dict(zip(WORKERS[::2], WORKERS[1::2]))
    groups = [flags["--security-group"], "sg-87654321"]
    statements = by_id(render("workers", *WORKERS, "--security-group", groups[1]))
    pool = PoolSpec(
        name=flags["--pool"], provider="cloud", model="qwen", checkpoint="/models/qwen",
        gpu_type="T4", image="vllm/vllm-openai:v0.10.2", max_replicas=1,
        options={"region": REGION, "instance_type": flags["--instance-type"], "ami_id": flags["--ami"],
                 "subnet_id": flags["--subnet"], "security_group_ids": groups},
    )
    provisioner = CloudProvisioner(flags["--namespace"])
    sdk = MagicMock()
    sdk.run_instances.return_value = {"Instances": [{"InstanceId": "i-example"}]}
    monkeypatch.setattr(provisioner, "_client", lambda region: sdk)
    provisioner._launch(pool, "1a" * 16)
    request = sdk.run_instances.call_args.kwargs

    ec2 = f"arn:aws:ec2:{REGION}:{ACCOUNT}"
    [interface] = request["NetworkInterfaces"]
    inputs = set(statements["UseApprovedWorkerInputs"]["Resource"])
    assert inputs == {f"arn:aws:ec2:{REGION}::image/{request['ImageId']}",
                      f"{ec2}:subnet/{interface['SubnetId']}",
                      *(f"{ec2}:security-group/{g}" for g in interface["Groups"])}
    assert request["MinCount"] == request["MaxCount"] == 1
    assert not interface["AssociatePublicIpAddress"]
    assert not {"IamInstanceProfile", "KeyName", "LaunchTemplate"} & request.keys()

    tag_on_create = statements["ApplyWorkerTagsOnlyAtLaunch"]
    [tag_spec] = request["TagSpecifications"]
    assert tag_spec["ResourceType"] == "instance"
    assert tag_on_create["Resource"] == f"{ec2}:instance/*"
    tags = {tag["Key"]: tag["Value"] for tag in tag_spec["Tags"]}
    context = {f"aws:RequestTag/{k}": v for k, v in tags.items()}
    context.update({"ec2:CreateAction": "RunInstances", "ec2:InstanceType": request["InstanceType"],
                    "ec2:MetadataHttpTokens": request["MetadataOptions"]["HttpTokens"],
                    "ec2:InstanceMarketType": "on-demand"})
    assert "InstanceMarketOptions" not in request
    for statement in (statements["LaunchNamedWorker"], tag_on_create):
        conditions = statement["Condition"]
        assert set(conditions["ForAllValues:StringEquals"]["aws:TagKeys"]) == set(tags)
        assert all(context[k] == v for k, v in conditions["StringEquals"].items())
        assert all(fnmatchcase(context[k], v) for k, v in conditions["StringLike"].items())
    assert tag_on_create["Condition"]["StringEquals"]["ec2:CreateAction"] == "RunInstances"
    assert statements["CreateWorkerDisksAtLaunch"]["Resource"] == f"{ec2}:volume/*"
    assert statements["CreatePrivateWorkerInterface"]["Resource"] == f"{ec2}:network-interface/*"
    assert statements["LaunchNamedWorker"]["Resource"] == f"{ec2}:instance/*"
    assert not {"ec2:CreateVolume", "ec2:CreateNetworkInterface", "iam:PassRole"} & {
        action for entry in statements.values() for action in (
            [entry["Action"]] if isinstance(entry["Action"], str) else entry["Action"]
        )
    }


@pytest.mark.parametrize("flag,bad", [
    ("--namespace", "*"), ("--ami", "ami-*"), ("--subnet", "*"),
    ("--security-group", "sg-*"), ("--instance-type", "g4dn.*"), ("--region", "*"),
])
def test_resource_inputs_cannot_broaden_grants(flag, bad):
    result = render("workers", *WORKERS, flag, bad)
    assert result.returncode != 0
    assert not result.stdout


RUN_NAME = "tf-test-1234567890abcdef"
LIVE_TEST = [
    "--namespace", RUN_NAME, "--ami", "ami-12345678", "--subnet", "subnet-12345678",
    "--security-group", "sg-12345678", "--guard-role", "TestCleanupRole",
]


def test_live_test_policy_matches_launch_and_cleanup_for_one_approved_namespace():
    policy = by_id(render("live-test", *LIVE_TEST, "--guard-stack", "TestGuard"))
    ec2_launch = policy["LaunchNamedWorker"]["Condition"]["StringEquals"]
    assert ec2_launch["aws:RequestTag/TokenFlowNamespace"] == RUN_NAME
    assert ec2_launch["aws:RequestTag/TokenFlowPool"] == "live-test"
    assert ec2_launch["ec2:InstanceType"] == "g4dn.xlarge"
    assert policy["TerminateOwnedWorkers"]["Condition"]["StringEquals"]["ec2:ResourceTag/TokenFlowNamespace"] == RUN_NAME
    assert policy["ReadCleanupStack"]["Resource"] == f"arn:aws:cloudformation:{REGION}:{ACCOUNT}:stack/TestGuard/*"
    assert set(policy["ReadCleanupStack"]["Action"]) == {"cloudformation:GetTemplate", "cloudformation:DescribeStacks"}
    assert policy["CheckCleanupRole"]["Resource"] == f"arn:aws:iam::{ACCOUNT}:role/TestCleanupRole"
    assert policy["CheckCleanupRole"]["Action"] == "iam:SimulatePrincipalPolicy"
    states = f"arn:aws:states:{REGION}:{ACCOUNT}"
    assert policy["ArmCleanup"]["Resource"] == f"{states}:stateMachine:TestGuard"
    assert set(policy["ArmCleanup"]["Action"]) == {"states:DescribeStateMachine", "states:StartExecution"}
    assert policy["VerifyAndFinishThisRun"]["Resource"] == f"{states}:execution:TestGuard:{RUN_NAME}"
    assert set(policy["VerifyAndFinishThisRun"]["Action"]) == {
        "states:DescribeExecution", "states:DescribeStateMachineForExecution",
        "states:GetExecutionHistory", "states:StopExecution",
    }


@pytest.mark.parametrize("flag,value", [
    ("--namespace", "tokenflow-demo"), ("--guard-stack", "Test*"), ("--guard-role", "*"), ("--region", "us-west-2"),
])
def test_guard_policy_rejects_ambiguous_resources(flag, value):
    result = render("live-test", *LIVE_TEST, flag, value)
    assert result.returncode != 0
    assert not result.stdout
