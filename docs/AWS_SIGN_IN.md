# Sign in to AWS from Windows

Use a short-lived session for local development. The control host in AWS should use an EC2 role. The SDK accepts both credential sources.

## Personal or federated AWS console sign-in

Use AWS CLI v2 with the `login` command. In **Windows PowerShell**, for the current-user installation:

```powershell
$Aws = "$env:LOCALAPPDATA\Programs\Amazon\AWSCLIV2\aws.exe"
& $Aws --version
& $Aws login --profile default --region us-east-1
& $Aws configure set region us-east-1 --profile default
& $Aws sts get-caller-identity --profile default
```

Sign in through the browser using a non-root identity with MFA. If your organization manages your IAM identity, its administrator can grant `SignInLocalDevelopmentAccess` for this login flow and separately grant the resource permissions needed for your deployment. The identity check returns the AWS account and principal. Keep session caches and keys outside the repo. [AWS console login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html), [AWS CLI installation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

## Organizational AWS access portal

For a standalone account, first check Billing → Credits for the type, amount, eligible services and expiry of any grant. Creating an AWS Organization makes the existing account its management account. AWS warns that creating an organization can expire Free Tier credits. AWS Activate is governed by its promotional and award-specific terms. For confirmation about a specific grant or coverage in the transition month, contact Account and Billing Support before enabling Organizations. [Create an organization](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_org_create.html), [Identity Center prerequisites](https://docs.aws.amazon.com/singlesignon/latest/userguide/enable-identity-center.html), [Activate terms](https://aws.amazon.com/activate/terms/), [Billing credit rules](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html)

If you are awaiting written confirmation about an existing Activate award, wait for Billing Support to confirm how creating the organization affects **that account, award and transition month** before enabling Organizations. After the account owner approves setup, enable AWS account access in the IAM Identity Center organization instance. In its default directory, choose **Users → Add user** and send the password setup email to the operator. Configure MFA under **Settings → Authentication → Multi-factor authentication**. Assign the scoped permission set to that user for the intended AWS account and wait for provisioning. The user sets a password and registers their MFA device. [AWS setup walkthrough](https://docs.aws.amazon.com/singlesignon/latest/userguide/quick-start-default-idc.html), [TokenFlow operator policy](AWS_OPERATOR.md)

If you normally choose an account and role in an IAM Identity Center portal, use this flow in place of `aws login`. Have your organization's start URL and SSO region ready. The SSO region can differ from the EC2 region.

```powershell
$Aws = "$env:LOCALAPPDATA\Programs\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path -LiteralPath $Aws)) { $Aws = (Get-Command aws.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source }
& $Aws configure sso --profile default
& $Aws sso login --profile default
& $Aws configure set region us-east-1 --profile default
& $Aws sts get-caller-identity --profile default --region us-east-1
```

Install TokenFlow with its `aws` extra in the Python environment that will run the gateway. This includes the SDK support for cached login credentials. Authentication can be verified without launching an instance. [Boto3 credential providers](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)

In the SSO wizard, enter a session name such as `tokenflow`, the access portal start URL displayed in Identity Center settings, the Identity Center region, and registration scope `sso:account:access`. Choose the account that owns the credits and its assigned `TokenFlowOperator` permission set. Select `us-east-1` for the default client region. Check that STS returns the expected 12-digit account and an `assumed-role/AWSReservedSSO_TokenFlowOperator_…` ARN before requesting quota. [AWS CLI SSO configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)

## Verify the account, credits and GPU quota

After SSO sign-in, install the `aws` extra and run this account check in a private terminal. It works before you have an AMI, a subnet or an allowed GPU allocation.

```powershell
$account = Read-Host "12-digit account ID that owns the grant"
tokenflow aws-status --expected-account $account --profile default --region us-east-1
```

Check the principal ARN and credential source, each account-owned promotion's expiry and settled and estimated balance, and the regional applied G/VT and P vCPU limits. The G/VT request history includes pending requests. A pending request for four vCPUs can appear alongside an applied limit of zero. The command only reads STS, Billing and Service Quotas. GPU availability and the effect of a future billing change require separate confirmation. [Billing credit fields](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_billing_GetCredits.html), [quota request history](https://docs.aws.amazon.com/servicequotas/2019-06-24/apireference/API_ListRequestedServiceQuotaChangeHistoryByQuota.html)

Inspect the ARN from `sts get-caller-identity`. If it ends in `:root`, clear that CLI session with `& $Aws logout --profile default` and configure the non-root profile. Confirm the role belongs to the account that owns the award.

In **AWS Billing → Credits**, open the award and check its status, expiration, eligible products and estimated amount remaining. The remaining estimate accounts for open bills and may lag recent usage. The Billing API can also return these fields for the exact account identified by STS. [Billing console credit details](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html), [GetCredits](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_billing_GetCredits.html)

## Request four total On-Demand G/VT vCPUs in us-east-1

One `g4dn.xlarge` needs four **unused** G/VT vCPUs. Quotas apply across the account in that region. An increase to four creates no instance and reserves no hardware. Use one operator for this request. Their assigned permission set needs the quota-request permission. [TokenFlow operator policy](AWS_OPERATOR.md), [EC2 On-Demand quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html)

Run this block in the same PowerShell session after the written grant confirmation and non-root sign-in above. Use the 12-digit account ID from the award. The command refreshes the account status before submitting.

```powershell
& {
    if ($account -notmatch '^\d{12}$') { throw 'Set $account to the account ID that owns the grant.' }
    $json = tokenflow aws-status --expected-account $account --profile default --region us-east-1
    if ($LASTEXITCODE -ne 0) { throw 'Account inspection failed. Resolve the error before proceeding.' }
    $status = $json | Out-String | ConvertFrom-Json -ErrorAction Stop
    $role = "^arn:aws:sts::${account}:assumed-role/AWSReservedSSO_TokenFlowOperator_[a-fA-F0-9]+/[^/]+$"
    if ($status.account -ne $account -or $status.region -ne 'us-east-1' -or
        $status.credential_source -ne 'sso' -or $status.principal_arn -cnotmatch $role) {
        throw 'Sign in with the TokenFlowOperator SSO permission set for the expected account.'
    }
    $quota = $status.on_demand_gpu_quotas.'G/VT'
    if ($quota.code -ne 'L-DB2E81BA' -or $null -eq $quota.applied_vcpus -or $null -eq $status.g_vt_quota_requests) {
        throw 'Missing G/VT quota or request history.'
    }
    if ($quota.applied_vcpus -ge 4) { Write-Host "Already applied: $($quota.applied_vcpus) total vCPUs."; return }
    $review = @($status.g_vt_quota_requests | Where-Object {
        $_.status -notin 'DENIED','NOT_APPROVED','CASE_CLOSED','INVALID_REQUEST','APPROVED' -or
        ($_.status -eq 'APPROVED' -and $_.requested_vcpus -ge 4)
    })
    if ($review.Count) { $review | Format-Table id,case_id,status,requested_vcpus,last_updated; return }
    $previousAttempts = $env:AWS_MAX_ATTEMPTS
    try {
        $env:AWS_MAX_ATTEMPTS = '1'
        & $Aws service-quotas request-service-quota-increase --profile default --region us-east-1 --service-code ec2 --quota-code L-DB2E81BA --desired-value 4 --output json --no-cli-pager
        if ($LASTEXITCODE -ne 0) { throw 'Submission outcome uncertain. Check request history before retrying.' }
    } finally { $env:AWS_MAX_ATTEMPTS = $previousAttempts }
}
```

Save the returned request ID. Use `tokenflow aws-status --expected-account $account --profile default --region us-east-1` to follow it. For `PENDING` or `CASE_OPENED`, follow the existing request. For `APPROVED` with an applied value below four, recheck the applied limit. If the discrepancy persists, ask Support about the displayed request or case ID. After a timeout or disconnect, check history and resolve any uncertainty with Service Quotas before submitting again. [Request a quota increase](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html)
