import json
import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize("flag,bad", [
    ("--namespace", "*"), ("--ami", "ami-*"), ("--subnet", "*"),
    ("--security-group", "sg-*"), ("--instance-type", "g4dn.*"), ("--region", "*"),
])
def test_resource_inputs_cannot_broaden_grants(flag, bad):
    result = render("workers", *WORKERS, flag, bad)
    assert result.returncode != 0
    assert not result.stdout
