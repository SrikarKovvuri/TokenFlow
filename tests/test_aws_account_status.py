from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
import json
import sys
from unittest.mock import MagicMock

import pytest

from tokenflow.cli import main
from tokenflow.provisioner.aws_account import account_status, G_VT_QUOTA, P_QUOTA


boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber

ACCOUNT = "123456789012"


def test_wrong_account_stops_before_billing_and_quota_access():
    session = MagicMock()
    session.client.return_value.get_caller_identity.return_value = {"Account": ACCOUNT}
    with pytest.raises(ValueError, match="authenticated AWS account differs"):
        account_status(session, "999999999999")
    session.client.assert_called_once_with("sts")


def test_cli_reports_credits_and_zero_applied_quota_through_sdk(monkeypatch, capsys):
    session = boto3.Session(aws_access_key_id="offline", aws_secret_access_key="offline", region_name="us-east-1")
    clients = {name: session.client(name) for name in ("sts", "billing", "service-quotas")}
    monkeypatch.setattr(session, "client", lambda name, **_: clients[name])
    make_session = MagicMock(return_value=session)
    monkeypatch.setattr(boto3, "Session", make_session)
    monkeypatch.setattr(sys, "argv", ["tokenflow", "aws-status", "--expected-account", ACCOUNT])
    now = datetime.now(UTC)
    def dollars(amount):
        return {"currencyCode": "USD", "currencyAmount": amount}

    grant = {
        "creditId": "private-id", "accountId": ACCOUNT, "creditType": "Promotion",
        "initialAmount": dollars("100"), "estimatedAmount": dollars("24"), "remainingAmount": dollars("30"),
        "creditStatus": "ENABLED", "description": "Test award", "startDate": now - timedelta(days=1),
        "endDate": now + timedelta(days=30), "applicableProductNames": ["Amazon Elastic Compute Cloud"],
    }
    arn = f"arn:aws:sts::{ACCOUNT}:assumed-role/AWSReservedSSO_TokenFlowOperator_example/session"
    with ExitStack() as stack:
        stubs = {name: stack.enter_context(Stubber(client)) for name, client in clients.items()}
        stubs["sts"].add_response("get_caller_identity", {"Arn": arn, "Account": ACCOUNT}, {})
        stubs["billing"].add_response("get_credits", {"credits": [grant, {**grant, "accountId": "999999999999"}]},
                                      {"accountId": ACCOUNT, "payerAccountFlag": False, "startDate": ANY})
        quotas = stubs["service-quotas"]
        for code in (G_VT_QUOTA, P_QUOTA):
            quotas.add_response("get_service_quota", {"Quota": {"Value": 0}}, {"ServiceCode": "ec2", "QuotaCode": code})
        lookup = {"ServiceCode": "ec2", "QuotaCode": G_VT_QUOTA}
        history = "list_requested_service_quota_change_history_by_quota"
        quotas.add_response(history, {"RequestedQuotas": [], "NextToken": "page2"}, lookup)
        quotas.add_response(history, {"RequestedQuotas": [{"Id": "example", "Status": "PENDING", "DesiredValue": 4}]},
                            {**lookup, "NextToken": "page2"})
        assert main() == 0
        make_session.assert_called_once_with(profile_name="default", region_name="us-east-1")
        for stub in stubs.values():
            stub.assert_no_pending_responses()
    output = capsys.readouterr().out
    result = json.loads(output)
    assert "private-id" not in output
    assert result["account"] == ACCOUNT and result["principal_arn"] == arn
    assert result["on_demand_gpu_quotas"]["G/VT"]["applied_vcpus"] == 0
    assert result["on_demand_gpu_quotas"]["P"]["applied_vcpus"] == 0
    assert result["g_vt_quota_requests"] == [{"id": "example", "status": "PENDING", "requested_vcpus": 4}]
    award, = result["credits"]
    assert award["expires"] == grant["endDate"].isoformat()
    assert award["remaining"] == dollars("30") and award["estimated_remaining"] == dollars("24")
    assert award["initial"] == dollars("100") and award["lists_ec2"]
    assert award["eligible_on_demand_ec2_usd"] == "24"
