"""Run one bounded, private GPU integration check in us-east-1."""

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_CEILING
import json
from ipaddress import IPv4Network
from pathlib import Path
import re
import secrets
import time

import httpx

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.provisioner.aws_account import G_VT_QUOTA, promotion_balance
from tokenflow.provisioner.base import PoolSpec
from tokenflow.provisioner.cloud import CloudProvisioner


REGION = "us-east-1"
INSTANCE_TYPE = "g4dn.xlarge"
MAX_USD = Decimal("10")
GUARD_TEMPLATE = Path(__file__).resolve().parents[1] / "deploy/aws-test-guard.yaml"


def inventory(client, action, field, **kwargs):
    for page in client.get_paginator(action).paginate(**kwargs):
        yield from page[field]


def price(pricing, unit, attributes):
    filters = [{"Type": "TERM_MATCH", "Field": key, "Value": value}
               for key, value in {"regionCode": REGION, **attributes}.items()]
    rates = set()
    products = inventory(pricing, "get_products", "PriceList", ServiceCode="AmazonEC2", Filters=filters)
    for product in products:
        for offer in json.loads(product)["terms"]["OnDemand"].values():
            for dimension in offer["priceDimensions"].values():
                if dimension["unit"] == unit and dimension["beginRange"] == "0":
                    rates.add(Decimal(dimension["pricePerUnit"]["USD"]))
    if len(rates) != 1 or min(rates) <= 0:
        raise ValueError(f"Expected one positive USD rate per {unit}, got {rates}")
    return rates.pop()


def verify_quota(ec2, quotas):
    used = Counter()
    for row in inventory(ec2, "describe_instances", "Reservations", Filters=[
        {"Name": "instance-type", "Values": ["g*", "vt*"]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ]):
        for vm in row["Instances"]:
            if not vm.get("InstanceLifecycle"):
                used[vm["InstanceType"]] += 1
    for reservation in inventory(ec2, "describe_capacity_reservations", "CapacityReservations",
                                 Filters=[{"Name": "state", "Values": ["active"]}]):
        kind = reservation["InstanceType"]
        if kind.startswith(("g", "vt")):
            used[kind] += reservation["AvailableInstanceCount"]
    types = sorted(set(used) | {INSTANCE_TYPE})
    cpu = {}
    for start in range(0, len(types), 100):
        for shape in ec2.describe_instance_types(InstanceTypes=types[start:start+100])["InstanceTypes"]:
            cpu[shape["InstanceType"]] = shape["VCpuInfo"]["DefaultVCpus"]
    quota = quotas.get_service_quota(ServiceCode="ec2", QuotaCode=G_VT_QUOTA)["Quota"]
    if "On-Demand G and VT" not in quota["QuotaName"]:
        raise ValueError(f"Unexpected EC2 quota {quota['QuotaName']}")
    in_use = sum(count * cpu[kind] for kind, count in used.items())
    if quota["Value"] < in_use + cpu[INSTANCE_TYPE]:
        raise ValueError(f"G/VT quota {quota['Value']} vCPUs, {in_use} already committed")
    return {"quota_vcpus": quota["Value"], "committed_vcpus": in_use}


def volume_size(image):
    if (image.get("PlatformDetails") != "Linux/UNIX" or image.get("ProductCodes")
            or image.get("RootDeviceType") != "ebs" or image.get("Architecture") != "x86_64"):
        raise ValueError("Provide an x86_64 Linux GPU AMI without software licensing charges")
    size = 0
    for mapping in image["BlockDeviceMappings"]:
        disk = mapping.get("Ebs")
        if not disk:
            continue
        if (not disk.get("DeleteOnTermination") or disk.get("VolumeType") != "gp3"
                or disk.get("Iops", 3000) > 3000 or disk.get("Throughput", 125) > 125):
            raise ValueError("AMI volumes must use baseline gp3 and DeleteOnTermination")
        size += disk["VolumeSize"]
    if not size:
        raise ValueError("No EBS volume size was available for pricing")
    return size


def check_network(ec2, args):
    subnet = ec2.describe_subnets(SubnetIds=[args.subnet])["Subnets"][0]
    if subnet.get("AssignIpv6AddressOnCreation"):
        raise ValueError("Choose a subnet without automatic IPv6 addresses")
    vpc_id = subnet["VpcId"]
    vpcs = ec2.describe_vpcs(VpcIds=[vpc_id]).get("Vpcs", [])
    if len(vpcs) != 1 or vpcs[0].get("VpcId") != vpc_id or vpcs[0].get("InstanceTenancy") != "default":
        raise ValueError("The shared-tenancy price estimate requires a worker VPC with verified default tenancy")
    offers = ec2.describe_instance_type_offerings(LocationType="availability-zone", Filters=[
        {"Name": "instance-type", "Values": [INSTANCE_TYPE]},
        {"Name": "location", "Values": [subnet["AvailabilityZone"]]},
    ])
    if not offers["InstanceTypeOfferings"]:
        raise ValueError(f"{INSTANCE_TYPE} is not offered in {subnet['AvailabilityZone']}")
    worker = ec2.describe_security_groups(GroupIds=[args.worker_group])["SecurityGroups"][0]
    if worker["VpcId"] != vpc_id:
        raise ValueError("The worker security group must belong to the worker VPC")
    if args.source_cidr:
        network = IPv4Network(args.source_cidr)
        if not any(network.subnet_of(IPv4Network(block)) for block in
                   ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
            raise ValueError("The VPN/control-host range must be a private IPv4 CIDR")
    allowed_groups = {args.source_group} if args.source_group else set()
    allowed_ranges = {args.source_cidr} if args.source_cidr else set()
    ports = set()
    for rule in worker["IpPermissions"]:
        sources = {source["GroupId"] for source in rule.get("UserIdGroupPairs", [])}
        ranges = {ip["CidrIp"] for ip in rule.get("IpRanges", [])}
        if (sources != allowed_groups or ranges != allowed_ranges or rule.get("Ipv6Ranges")
                or rule.get("PrefixListIds") or rule["IpProtocol"] != "tcp"
                or not 8000 <= rule["FromPort"] <= rule["ToPort"] <= 8001):
            raise ValueError("Worker ingress must allow only TCP 8000/8001 from the specified trusted source")
        ports.update(range(rule["FromPort"], rule["ToPort"] + 1))
    if ports != {8000, 8001}:
        raise ValueError("Both worker and GPU metrics ports are required")


def preflight(session, args):
    if args.execute and not args.expected_account:
        raise ValueError("Pass --expected-account with the 12-digit account approved for this run")
    identity = session.client("sts").get_caller_identity()
    if args.expected_account and identity["Account"] != args.expected_account:
        raise ValueError("The authenticated AWS account differs from --expected-account")
    ec2 = session.client("ec2")
    balance = promotion_balance(session.client("billing"), identity["Account"], args.credit_description)
    result = verify_quota(ec2, session.client("service-quotas"))
    check_network(ec2, args)
    image = ec2.describe_images(ImageIds=[args.ami])["Images"]
    if len(image) != 1 or image[0]["State"] != "available":
        raise ValueError("GPU AMI is unavailable")
    gib = volume_size(image[0])
    pricing = session.client("pricing")
    hourly = price(pricing, "Hrs", {
        "instanceType": INSTANCE_TYPE, "operatingSystem": "Linux", "tenancy": "Shared",
        "preInstalledSw": "NA", "capacitystatus": "Used", "operation": "RunInstances",
    })
    storage = price(pricing, "GB-Mo", {"volumeApiName": "gp3", "productFamily": "Storage"})
    # Two hours include boot and time for cloud-side termination after the
    # 30-minute deadline. Allow $2 for the control workflow and transfer.
    estimate = (2 * hourly + Decimal(2 * gib) * storage / (28 * 24) + Decimal("2")).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    if estimate > MAX_USD:
        raise ValueError(f"Estimated run {estimate:.2f} USD exceeds {MAX_USD} USD")
    if balance < estimate:
        raise ValueError(f"Eligible EC2 promotional balance {balance:.2f} USD is below estimate {estimate:.2f} USD")
    return {**result, "account": identity["Account"], "principal_is_root": identity["Arn"].endswith(":root"),
            "region": REGION, "instance": INSTANCE_TYPE,
            "hourly_usd": str(hourly), "ebs_gib": gib, "estimated_usd": f"{estimate:.2f}",
            "ebs_volumes": sum("Ebs" in m for m in image[0]["BlockDeviceMappings"]),
            "eligible_promotion_usd": str(balance), "credit_source": "billing:GetCredits"}


def verify_guard(cfn, sfn, iam, stack_name, namespace, account):
    """Only use the operator-installed guard defined by this checkout."""
    body = cfn.get_template(StackName=stack_name, TemplateStage="Original")["TemplateBody"]
    if not isinstance(body, str) or body.replace("\r\n", "\n").strip() != GUARD_TEMPLATE.read_text().strip():
        raise ValueError("Guard stack template differs from deploy/aws-test-guard.yaml")
    stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    if stack["StackStatus"] not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}:
        raise ValueError(f"Cleanup stack is not ready: {stack['StackStatus']}")
    machine = next(item["OutputValue"] for item in stack["Outputs"] if item["OutputKey"] == "Machine")
    state = sfn.describe_state_machine(stateMachineArn=machine)
    if state["status"] != "ACTIVE" or state["type"] != "STANDARD":
        raise ValueError("Cleanup guard must be an active Standard workflow")
    definition = guard_definition()
    if json.loads(state["definition"]) != definition:
        raise ValueError("Deployed cleanup workflow differs from the checked-in guard definition")
    decision = iam.simulate_principal_policy(
        PolicySourceArn=state["roleArn"], ActionNames=["ec2:TerminateInstances"],
        ResourceArns=[f"arn:aws:ec2:{REGION}:{account}:instance/i-00000000000000000"],
        ContextEntries=[
            {"ContextKeyName": "ec2:ResourceTag/TokenFlowNamespace", "ContextKeyValues": [namespace],
             "ContextKeyType": "string"},
            {"ContextKeyName": "ec2:ResourceTag/TokenFlowPool", "ContextKeyValues": ["live-test"],
             "ContextKeyType": "string"},
        ],
    )["EvaluationResults"][0]
    if decision["EvalDecision"] != "allowed" or decision.get("MissingContextValues"):
        raise ValueError("Cleanup role is missing tagged-instance termination permission")
    return machine, state["roleArn"]


def guard_definition():
    import yaml

    class TemplateLoader(yaml.SafeLoader):
        pass

    # CloudFormation uses these scalar tags outside the state-machine definition.
    # Preserve them as data when reading the trusted local template.
    for tag in ("!Ref", "!Sub", "!GetAtt"):
        TemplateLoader.add_constructor(tag, lambda loader, node: {node.tag: loader.construct_scalar(node)})
    template = yaml.load(GUARD_TEMPLATE.read_text(), Loader=TemplateLoader)
    return template["Resources"]["Guard"]["Properties"]["Definition"]


def arm_guard(sfn, machine, role, namespace):
    deadline = datetime.now(UTC) + timedelta(minutes=30)
    execution = sfn.start_execution(stateMachineArn=machine, name=namespace, input=json.dumps({
        # Independent invocations for one approved name must not share the
        # same execution, even if they start within the same second.
        "attempt": secrets.token_hex(16),
        "deadline": deadline.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "verify_until": (deadline + timedelta(minutes=10)).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }))["executionArn"]
    try:
        verify_guard_execution(sfn, execution, machine, role)
    except Exception:
        # StartExecution succeeded and RunInstances has not been called. This
        # invocation owns the execution and can safely stop it on a failed check.
        try:
            if sfn.describe_execution(executionArn=execution)["status"] == "RUNNING":
                sfn.stop_execution(executionArn=execution, cause="Pre-launch guard verification failed")
        except Exception as stop_error:
            print(f"Could not confirm pre-launch guard cleanup for {execution}: {stop_error}", flush=True)
        raise
    return execution, deadline


def verify_guard_execution(sfn, execution, machine, role):
    for _ in range(60):
        if sfn.describe_execution(executionArn=execution)["status"] != "RUNNING":
            break
        events = sfn.get_execution_history(executionArn=execution, reverseOrder=True, maxResults=20)["events"]
        if any(e["type"] == "WaitStateEntered" and e["stateEnteredEventDetails"]["name"] == "WaitForDeadline"
               for e in events):
            # Capture the definition and role for this specific execution. A
            # change between inspecting the machine and starting it must not
            # allow an instance to launch under an unverified deadline.
            snapshot = sfn.describe_state_machine_for_execution(executionArn=execution)
            if (snapshot["stateMachineArn"] != machine or snapshot["roleArn"] != role
                    or json.loads(snapshot["definition"]) != guard_definition()):
                raise ValueError("Active cleanup execution differs from the verified workflow or role")
            return
        time.sleep(2)
    raise RuntimeError(f"Cleanup guard did not reach its deadline wait: {execution}")


class AdoptOnly(CloudProvisioner):
    async def launch(self, pool, key):
        raise RuntimeError("Live validation permits exactly one EC2 launch")


def attachments(ec2, instance_id):
    instances = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"]
    disks = [m["Ebs"]["VolumeId"] for m in instances[0]["BlockDeviceMappings"] if "Ebs" in m]
    interfaces = [i["NetworkInterfaceId"] for i in instances[0]["NetworkInterfaces"]]
    return disks, interfaces


def verify_released(ec2, disks, interfaces, expected_disks):
    # This run launches exactly one ENI and an EBS-backed AMI. EC2 can return
    # incomplete inventory during transitions. An unknown volume is not proof
    # that all the volumes were deleted.
    if expected_disks < 1 or len(set(disks)) != expected_disks or len(interfaces) != 1:
        raise RuntimeError("EC2 attachment inventory was incomplete; release could not be fully verified")
    for _ in range(30):
        volumes = ec2.describe_volumes(Filters=[{"Name": "volume-id", "Values": disks}])["Volumes"]
        nics = ec2.describe_network_interfaces(
            Filters=[{"Name": "network-interface-id", "Values": interfaces}]
        )["NetworkInterfaces"]
        if not volumes and not nics:
            return
        time.sleep(5)
    raise RuntimeError(f"EC2 terminated but attached storage/interfaces still visible: {disks}, {interfaces}")


async def release_worker(ec2, provider, pool, vm, expected_disks, out):
    try:
        disks, interfaces = attachments(ec2, vm.external_id)
    finally:
        await provider.terminate(pool, vm)
        ec2.get_waiter("instance_terminated").wait(
            InstanceIds=[vm.external_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60}
        )
    verify_released(ec2, disks, interfaces, expected_disks)
    (out / "cleanup.json").write_text(json.dumps({
        "instance_id": vm.external_id, "volume_ids": disks, "interface_ids": interfaces,
        "release_verified_at": datetime.now(UTC).isoformat(),
    }, indent=2))


async def recover_launch(ec2, provider, pool, key, expected_disks, out):
    # RunInstances can succeed before its HTTP reply is lost. Inventory can
    # also lag. The independent deadline guard covers a longer visibility delay.
    for delay in (0, 2, 5, 10):
        if delay:
            await asyncio.sleep(delay)
        for vm in await provider.discover(pool):
            if vm.key == key:
                await release_worker(ec2, provider, pool, vm, expected_disks, out)
                return


async def check_gateway(pool, namespace, out, provider):
    catalog = {"namespace": namespace, "state_path": str(out / "control.sqlite3"),
               "interval_seconds": 3, "startup_timeout_seconds": 1500,
               "pools": [pool.model_dump()]}
    path = out / "platform.json"
    path.write_text(json.dumps(catalog))
    api_key, admin_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    app = create_app(Settings(platform_config=path, api_keys=[api_key], admin_key=admin_key),
                     providers={"cloud": provider})
    headers = {"Authorization": f"Bearer {admin_key}"}
    async with asyncio.timeout(20 * 60), app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway",
                                     headers={"Authorization": f"Bearer {api_key}"}) as client:
            while True:
                workers = (await client.get("/v1/workers", headers=headers)).raise_for_status().json()
                if len(workers) == 1 and workers[0]["health_status"] == "healthy":
                    telemetry = workers[0].get("telemetry") or {}
                    if all(telemetry.get(key) is not None for key in ("running", "waiting", "kv_cache_usage", "gpu_utilization")):
                        break
                await asyncio.sleep(3)
            payload = {"model": pool.model, "messages": [{"role": "user", "content": "Say hello."}],
                       "temperature": 0, "max_tokens": 12}
            chat = (await client.post("/v1/chat/completions", json=payload)).raise_for_status().json()
            if not chat["choices"][0]["message"]["content"]:
                raise RuntimeError("Worker returned an empty chat completion")
            async with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as response:
                response.raise_for_status()
                events = [line[6:] async for line in response.aiter_lines() if line.startswith("data: ")]
            if not events or events[-1] != "[DONE]" or not any(
                choice.get("delta", {}).get("content")
                for event in events[:-1] for choice in json.loads(event)["choices"]
            ):
                raise RuntimeError("Incomplete streaming chat response")
            return {"worker": workers[0], "chat": chat, "stream": events}


async def run(args):
    import boto3

    session = boto3.Session(region_name=REGION)
    report = preflight(session, args)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    if report["principal_is_root"]:
        raise ValueError("Use a least-privilege IAM role or IAM user before executing a GPU test")
    if not args.confirm_private_route:
        raise ValueError("Confirm an existing private route from this host to the worker ports before --execute")
    if not args.local_checkpoint and not args.confirm_artifact_egress:
        raise ValueError("Confirm budgeted artifact egress or use a pre-staged local checkpoint and image")
    if args.local_checkpoint and not args.local_checkpoint.startswith("/"):
        raise ValueError("--local-checkpoint must be an absolute path on the AMI")
    if args.image and not args.local_checkpoint:
        raise ValueError("Overriding the image requires pre-staged assets")
    namespace = args.namespace if args.namespace is not None else "tf-test-" + secrets.token_hex(8)
    if not re.fullmatch(r"tf-test-[0-9a-f]{16}", namespace):
        raise ValueError("Expected --namespace tf-test- followed by 16 lowercase hexadecimal characters")
    out = (args.output / namespace).resolve()
    out.mkdir(parents=True)
    print(f"Run directory {out}; cleanup execution {namespace}", flush=True)
    pool = PoolSpec(
        name="live-test", provider="cloud", backend="vllm", model="qwen-test", gpu_type="T4",
        gpu_count=1, min_replicas=1, max_replicas=1, capacity=1, port=8000,
        checkpoint=args.local_checkpoint or "Qwen/Qwen2.5-0.5B-Instruct",
        image=args.image or "vllm/vllm-openai:v0.10.2",
        options={"region": REGION, "instance_type": INSTANCE_TYPE, "ami_id": args.ami,
                 "subnet_id": args.subnet, "security_group_ids": [args.worker_group],
                 "dtype": "float16", "max_model_len": 512,
                 "image_pull_policy": "never" if args.local_checkpoint else "missing"},
    )
    cfn, sfn, ec2 = (session.client(service) for service in ("cloudformation", "stepfunctions", "ec2"))
    provider = CloudProvisioner(namespace, session=session)
    machine, role = verify_guard(cfn, sfn, session.client("iam"), args.guard_stack, namespace, report["account"])
    guard, deadline = arm_guard(sfn, machine, role, namespace)
    key = secrets.token_hex(16)
    report.update(namespace=namespace, launch_key=key, guard=guard, deadline=deadline.isoformat())
    (out / "run.json").write_text(json.dumps(report, indent=2))
    if datetime.now(UTC) >= deadline - timedelta(minutes=22):
        raise RuntimeError(f"Insufficient cleanup window; guard left armed: {guard}")
    # There is one caller and one launch key. Adoption exercises normal
    # discovery and HTTP routing without authorizing any replacement instance.
    try:
        vm = await provider.launch(pool, key)
    except (Exception, asyncio.CancelledError) as launch_error:
        try:
            await recover_launch(ec2, provider, pool, key, report["ebs_volumes"], out)
        except Exception as cleanup_error:
            launch_error.add_note(f"Could not confirm immediate cleanup: {cleanup_error}")
        launch_error.add_note(f"Launch outcome was uncertain. Independent cleanup guard remains armed: {guard}")
        raise
    try:
        report["instance_id"] = vm.external_id
        (out / "run.json").write_text(json.dumps(report, indent=2))
        print(f"Allocated {vm.external_id}, guard expires {deadline.isoformat()}", flush=True)
        result = await check_gateway(pool, namespace, out, AdoptOnly(namespace, session=session))
        (out / "responses.json").write_text(json.dumps(result, indent=2))
    finally:
        await release_worker(ec2, provider, pool, vm, report["ebs_volumes"], out)
        if sfn.describe_execution(executionArn=guard)["status"] == "RUNNING":
            sfn.stop_execution(executionArn=guard, cause="EC2 termination confirmed")
        print(f"Confirmed instance termination: {vm.external_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", help="fresh namespace preapproved in a live-test policy (default: random)")
    parser.add_argument("--expected-account", help="12-digit credit-owning account ID, required for --execute")
    parser.add_argument("--ami", required=True, help="GPU-ready Linux AMI with Docker and NVIDIA container runtime")
    parser.add_argument("--subnet", required=True)
    parser.add_argument("--worker-group", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-group", help="trusted control host's security group")
    source.add_argument("--source-cidr", help="trusted private VPN or control-host IPv4 CIDR")
    parser.add_argument("--confirm-private-route", action="store_true",
                        help="the launcher has a tested private route to this subnet and worker ports")
    parser.add_argument("--confirm-artifact-egress", action="store_true",
                        help="this subnet can download the fixed image/checkpoint within the $2 allowance")
    parser.add_argument("--local-checkpoint", help="use Qwen2.5-0.5B-Instruct already present on the AMI")
    parser.add_argument("--image", help="OCI image already loaded on an AMI with a local checkpoint")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/aws-live"))
    parser.add_argument("--guard-stack", default="tokenflow-gpu-test-guard")
    parser.add_argument("--credit-description", help="optionally restrict credit to one exact promotional award description")
    parser.add_argument("--execute", action="store_true", help="arm the cleanup guard and run the live check")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
