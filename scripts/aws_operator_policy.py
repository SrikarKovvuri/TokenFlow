"""Render a TokenFlow operator's IAM Identity Center inline policy locally."""

import argparse
import json
import re
from pathlib import Path


QUOTA_CODE = "L-DB2E81BA"  # Running On-Demand G and VT instances, measured in vCPUs
GPU_TAGS = ["TokenFlowNamespace", "TokenFlowPool", "TokenFlowKey", "Name"]


def checked(value, pattern, label):
    if not re.fullmatch(pattern, value, re.ASCII):
        raise ValueError(f"Invalid {label}. Supply an exact value without IAM wildcards.")
    return value


def allow(sid, actions, resources, condition=None):
    statement = {"Sid": sid, "Effect": "Allow", "Action": actions, "Resource": resources}
    if condition:
        statement["Condition"] = condition
    return statement


def bootstrap_policy(account, region, allow_quota_request=False):
    checked(account, r"[0-9]{12}", "12-digit AWS account ID")
    checked(region, r"(?:us|eu|ap|af|ca|il|me|mx|sa)-[a-z]+(?:-[a-z]+)?-[1-9][0-9]*", "AWS Region")
    if region.startswith("us-gov-"):
        raise ValueError("This policy targets the commercial AWS partition")
    quota = f"arn:aws:servicequotas:{region}:{account}:ec2/{QUOTA_CODE}"
    regional = {"StringEquals": {"aws:RequestedRegion": region}}
    quota_actions = ["servicequotas:GetServiceQuota", "servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota"]
    if allow_quota_request:
        quota_actions.append("servicequotas:RequestServiceQuotaIncrease")
    return {"Version": "2012-10-17", "Statement": [
        allow("VerifySession", "sts:GetCallerIdentity", "*"),
        allow("ReadCreditsAndPrices", ["billing:GetCredits", "pricing:GetProducts"], "*",
              {"StringEquals": {"aws:RequestedRegion": "us-east-1"}}),
        allow("InspectRegionalGpuInfrastructure", [
            "ec2:DescribeInstances", "ec2:DescribeInstanceStatus", "ec2:DescribeCapacityReservations",
            "ec2:DescribeImages", "ec2:DescribeInstanceTypes", "ec2:DescribeInstanceTypeOfferings",
            "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups", "ec2:DescribeVolumes",
            "ec2:DescribeNetworkInterfaces",
        ], "*", regional),
        allow("ReadRegionalMetrics", [
            "cloudwatch:ListMetrics", "cloudwatch:GetMetricData", "cloudwatch:GetMetricStatistics",
        ], "*", regional),
        allow("RegionalGpuQuota", quota_actions, quota, regional),
    ]}


def worker_statements(account, region, namespace, pool, ami, subnet, groups, instance_type):
    checked(namespace, r"[a-z][a-z0-9-]{0,31}", "TokenFlow namespace")
    checked(pool, r"[a-z][a-z0-9-]{0,31}", "TokenFlow pool name")
    id_suffix = r"(?:[0-9a-f]{8}|[0-9a-f]{17})"
    checked(ami, "ami-" + id_suffix, "AMI ID")
    checked(subnet, "subnet-" + id_suffix, "subnet ID")
    if not groups:
        raise ValueError("At least one approved worker security group is required")
    for group in groups:
        checked(group, "sg-" + id_suffix, "security group ID")
    checked(instance_type, r"[a-z0-9-]+\.[a-z0-9]+", "instance type")

    ec2 = f"arn:aws:ec2:{region}:{account}"
    subnet_arn = f"{ec2}:subnet/{subnet}"
    instance_arn = f"{ec2}:instance/*"
    instance_tags = {
        "aws:RequestTag/TokenFlowNamespace": namespace,
        "aws:RequestTag/TokenFlowPool": pool,
        "aws:RequestTag/Name": f"tokenflow-{pool}",
    }
    return [
        allow("UseApprovedWorkerInputs", "ec2:RunInstances", [
            f"arn:aws:ec2:{region}::image/{ami}", subnet_arn,
            *(f"{ec2}:security-group/{group}" for group in sorted(set(groups))),
        ]),
        allow("CreatePrivateWorkerInterface", "ec2:RunInstances", f"{ec2}:network-interface/*", {
            "ArnEquals": {"ec2:Subnet": subnet_arn},
            "Bool": {"ec2:AssociatePublicIpAddress": "false"},
        }),
        allow("CreateWorkerDisksAtLaunch", "ec2:RunInstances", f"{ec2}:volume/*"),
        allow("LaunchNamedWorker", "ec2:RunInstances", instance_arn, {
            "StringEquals": {**instance_tags, "ec2:InstanceType": instance_type,
                             "ec2:InstanceMarketType": "on-demand", "ec2:MetadataHttpTokens": "required"},
            "StringLike": {"aws:RequestTag/TokenFlowKey": "?" * 32},
            "ForAllValues:StringEquals": {"aws:TagKeys": GPU_TAGS},
        }),
        allow("ApplyWorkerTagsOnlyAtLaunch", "ec2:CreateTags", instance_arn, {
            "StringEquals": {"ec2:CreateAction": "RunInstances", **instance_tags},
            "StringLike": {"aws:RequestTag/TokenFlowKey": "?" * 32},
            "ForAllValues:StringEquals": {"aws:TagKeys": GPU_TAGS},
        }),
        allow("TerminateOwnedWorkers", "ec2:TerminateInstances", instance_arn, {
            "StringEquals": {"ec2:ResourceTag/TokenFlowNamespace": namespace,
                             "ec2:ResourceTag/TokenFlowPool": pool},
            "StringLike": {"ec2:ResourceTag/TokenFlowKey": "?" * 32},
        }),
    ]


def live_test_guard_statements(account, region, namespace, stack, role):
    if region != "us-east-1":
        raise ValueError("The guarded GPU test runs in us-east-1")
    checked(namespace, r"tf-test-[0-9a-f]{16}", "live-test namespace")
    checked(stack, r"[A-Za-z][A-Za-z0-9-]{0,79}", "cleanup stack and state machine name")
    checked(role, r"[A-Za-z0-9_+=,.@-]{1,64}", "cleanup IAM role name")
    states = f"arn:aws:states:{region}:{account}"
    return [
        allow("ReadCleanupStack", ["cloudformation:DescribeStacks", "cloudformation:GetTemplate"],
              f"arn:aws:cloudformation:{region}:{account}:stack/{stack}/*"),
        allow("CheckCleanupRole", "iam:SimulatePrincipalPolicy", f"arn:aws:iam::{account}:role/{role}"),
        allow("ArmCleanup", ["states:DescribeStateMachine", "states:StartExecution"],
              f"{states}:stateMachine:{stack}"),
        allow("VerifyAndFinishThisRun", ["states:DescribeExecution", "states:DescribeStateMachineForExecution",
                                         "states:GetExecutionHistory", "states:StopExecution"],
              f"{states}:execution:{stack}:{namespace}"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    bootstrap = modes.add_parser("bootstrap", help="inspect inventory, price, credit and G/VT quota")
    workers = modes.add_parser("workers", help="add launches and termination for one approved GPU pool")
    live_test = modes.add_parser("live-test", help="limit the guarded T4 test to one deployment namespace")
    for mode in (bootstrap, workers, live_test):
        mode.add_argument("--account-id", required=True)
        mode.add_argument("--region", default="us-east-1")
        mode.add_argument("--allow-quota-request", action="store_true")
        mode.add_argument("--output", type=Path, help="write UTF-8 JSON to this local file (default: stdout)")
    for mode in (workers, live_test):
        mode.add_argument("--namespace", required=True)
        mode.add_argument("--ami", required=True)
        mode.add_argument("--subnet", required=True)
        mode.add_argument("--security-group", action="append", required=True)
    workers.add_argument("--pool", required=True)
    workers.add_argument("--instance-type", required=True)
    live_test.set_defaults(pool="live-test", instance_type="g4dn.xlarge")
    live_test.add_argument("--guard-stack", default="tokenflow-gpu-test-guard")
    live_test.add_argument("--guard-role", required=True, help="physical IAM role name created by the cleanup stack")
    args = parser.parse_args()
    try:
        policy = bootstrap_policy(args.account_id, args.region, args.allow_quota_request)
        if args.mode in {"workers", "live-test"}:
            policy["Statement"].extend(worker_statements(
                args.account_id, args.region, args.namespace, args.pool,
                args.ami, args.subnet, args.security_group, args.instance_type,
            ))
        if args.mode == "live-test":
            policy["Statement"].extend(live_test_guard_statements(
                args.account_id, args.region, args.namespace, args.guard_stack, args.guard_role,
            ))
    except ValueError as exc:
        parser.error(str(exc))
    data = json.dumps(policy, indent=2) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
