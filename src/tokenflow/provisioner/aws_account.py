"""Read account-owned promotional credits and regional GPU quotas."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal


G_VT_QUOTA = "L-DB2E81BA"
P_QUOTA = "L-417A185B"


def promotions(billing, account, now):
    response = billing.get_credits(
        accountId=account, payerAccountFlag=False, startDate=now - timedelta(days=364),
    )
    return [credit for credit in response["credits"] if credit["accountId"] == account
            and credit["creditType"].casefold().startswith("promo")]


def eligible_ec2_usd(credit, now):
    """Account for expiry, purchase restrictions, currency and unsettled usage."""
    through = now + timedelta(hours=2)
    if (credit.get("creditStatus") != "ENABLED" or credit["startDate"] > now
            or (credit.get("exhaustDate") and credit["exhaustDate"] <= through)
            or (credit.get("endDate") and credit["endDate"] < through)
            or "Amazon Elastic Compute Cloud" not in credit.get("applicableProductNames", [])):
        return Decimal(0)
    purchases = {p.lower().replace("-", "").replace("_", "").replace(" ", "")
                 for p in credit.get("purchaseTypeApplications", [])}
    if purchases and purchases.isdisjoint({"all", "ondemand", "ondemandinstances"}):
        return Decimal(0)
    amounts = [credit["remainingAmount"]]
    if credit.get("estimatedAmount") is not None:
        amounts.append(credit["estimatedAmount"])
    if any(amount["currencyCode"] != "USD" for amount in amounts):
        return Decimal(0)
    return max(Decimal(0), min(Decimal(amount["currencyAmount"]) for amount in amounts))


def promotion_balance(billing, account, description=None):
    now = datetime.now(UTC)
    return sum((eligible_ec2_usd(credit, now) for credit in promotions(billing, account, now)
                if description is None or credit.get("description") == description), Decimal(0))


def account_status(session, expected_account):
    identity = session.client("sts").get_caller_identity()
    if identity["Account"] != expected_account:
        raise ValueError("The authenticated AWS account differs from --expected-account")

    now = datetime.now(UTC)
    billing = session.client("billing", region_name="us-east-1")
    awards = []
    for credit in promotions(billing, expected_account, now):
        awards.append({
            "description": credit.get("description"), "status": credit.get("creditStatus"),
            "expires": credit["endDate"].isoformat() if credit.get("endDate") else None,
            "initial": credit["initialAmount"],
            "remaining": credit["remainingAmount"], "estimated_remaining": credit.get("estimatedAmount"),
            "lists_ec2": "Amazon Elastic Compute Cloud" in credit.get("applicableProductNames", []),
            "purchase_types": credit.get("purchaseTypeApplications"),
            "eligible_on_demand_ec2_usd": str(eligible_ec2_usd(credit, now)),
        })

    quotas = session.client("service-quotas")
    applied = {}
    for family, code in (("G/VT", G_VT_QUOTA), ("P", P_QUOTA)):
        quota = quotas.get_service_quota(ServiceCode="ec2", QuotaCode=code)["Quota"]
        applied[family] = {"code": code, "applied_vcpus": quota["Value"]}
    history = quotas.get_paginator("list_requested_service_quota_change_history_by_quota")
    requests = []
    for page in history.paginate(ServiceCode="ec2", QuotaCode=G_VT_QUOTA):
        for item in page["RequestedQuotas"]:
            request = {"id": item["Id"], "status": item["Status"], "requested_vcpus": item["DesiredValue"]}
            if item.get("CaseId"):
                request["case_id"] = item["CaseId"]
            for field, name in (("Created", "created"), ("LastUpdated", "last_updated")):
                if item.get(field):
                    request[name] = item[field].isoformat()
            requests.append(request)
    return {
        "account": expected_account, "principal_arn": identity["Arn"],
        "credential_source": session.get_credentials().method, "region": session.region_name,
        "credits": awards, "on_demand_gpu_quotas": applied, "g_vt_quota_requests": requests,
    }
