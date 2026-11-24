import asyncio
import importlib.util
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_estimate_rounds_up_and_no_overbudget_launch(monkeypatch):
    session = MagicMock()
    session.client.return_value.describe_images.return_value = {"Images": [{"State": "available"}]}
    monkeypatch.setattr(live, "verify_quota", lambda *_: {"quota_vcpus": 4, "committed_vcpus": 0})
    monkeypatch.setattr(live, "promotion_balance", lambda *_: Decimal(50))
    monkeypatch.setattr(live, "check_network", lambda *_: None)
    monkeypatch.setattr(live, "volume_size", lambda *_: 60)
    monkeypatch.setattr(live, "price", lambda _, unit, __: Decimal("0.526") if unit == "Hrs" else Decimal("0.08"))
    assert live.preflight(session, MagicMock())["estimated_usd"] == "3.07"
    monkeypatch.setattr(live, "price", lambda *_: Decimal(20))
    with pytest.raises(ValueError, match="exceeds"):
        live.preflight(session, MagicMock())


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
        live.arm_guard(sfn, "machine", "run")
    with pytest.raises(RuntimeError, match="one EC2 launch"):
        asyncio.run(live.AdoptOnly("run").launch(None, "new-key"))


def test_guard_verifies_role_can_terminate_before_gpu_launch():
    cfn, sfn, iam = MagicMock(), MagicMock(), MagicMock()
    cfn.get_template.return_value = {"TemplateBody": live.GUARD_TEMPLATE.read_text()}
    cfn.describe_stacks.return_value = {"Stacks": [{"StackStatus": "CREATE_COMPLETE",
                                                    "Outputs": [{"OutputKey": "Machine", "OutputValue": "machine"}]}]}
    sfn.describe_state_machine.return_value = {"type": "STANDARD", "status": "ACTIVE", "roleArn": "role"}
    iam.simulate_principal_policy.return_value = {"EvaluationResults": [{"EvalDecision": "implicitDeny"}]}
    with pytest.raises(ValueError, match="permission"):
        live.verify_guard(cfn, sfn, iam, "stack", "tf-test-0123456789abcdef", "123456789012")


def test_private_vpn_ingress_allows_a_preloaded_worker_without_nat():
    ec2 = MagicMock()
    ec2.describe_subnets.return_value = {"Subnets": [{"VpcId": "vpc", "AvailabilityZone": "zone"}]}
    ec2.describe_instance_type_offerings.return_value = {"InstanceTypeOfferings": [{}]}
    rule = {"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8001, "IpRanges": [{"CidrIp": "10.0.7.9/32"}]}
    ec2.describe_security_groups.return_value = {"SecurityGroups": [{"VpcId": "vpc", "IpPermissions": [rule]}]}
    args = SimpleNamespace(subnet="subnet", worker_group="worker", source_group=None, source_cidr="10.0.7.9/32")
    live.check_network(ec2, args)
    ec2.describe_nat_gateways.assert_not_called()
    rule["IpRanges"].append({"CidrIp": "0.0.0.0/0"})
    with pytest.raises(ValueError, match="ingress"):
        live.check_network(ec2, args)
