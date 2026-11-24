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

After the account owner approves setup, use the IAM Identity Center organization instance with AWS account access in the chosen SSO region. In its default directory, choose **Users → Add user** and send the password setup email to the operator. Configure MFA under **Settings → Authentication → Multi-factor authentication**. Create the scoped permission set, assign it to that user for the intended AWS account and wait for provisioning. The user opens their invitation, sets a password and registers their MFA device themselves. [AWS setup walkthrough](https://docs.aws.amazon.com/singlesignon/latest/userguide/quick-start-default-idc.html), [TokenFlow operator policy](AWS_OPERATOR.md)

If you normally choose an account and role in an IAM Identity Center portal, use this flow in place of `aws login`. Have your organization's start URL and SSO region ready. The SSO region can differ from the EC2 region.

```powershell
& $Aws configure sso --profile default
& $Aws sso login --profile default
& $Aws configure set region us-east-1 --profile default
& $Aws sts get-caller-identity --profile default
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

In **Windows PowerShell**, check the account's applied regional On-Demand G/VT limit:

```powershell
& $Aws service-quotas get-service-quota --profile default --region us-east-1 --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.{Name:QuotaName,vCPUs:Value}'
```

One `g4dn.xlarge` needs four available vCPUs in this quota. The credit balance and the quota are independently administered. A quota increase request creates no instances, and the new limit takes effect only after AWS approves and applies it. [EC2 On-Demand quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html)
