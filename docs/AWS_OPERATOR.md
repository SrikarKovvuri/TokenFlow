# TokenFlow operator access in AWS

Start with **bootstrap** to inspect credits, GPU quota, prices and regional inventory. Add **workers** when an administrator has approved a specific AMI, instance type, private subnet, security group and TokenFlow pool. The renderer writes an IAM inline policy locally. An Identity Center administrator assigns the permission set to the one AWS account that owns the workload.

## 1. Render the initial permission set

In **Windows PowerShell**, from the TokenFlow repo root:

```powershell
$account = Read-Host "12-digit AWS account ID shown in your access portal"
$policy = Join-Path $env:TEMP "tokenflow-operator.json"
python scripts/aws_operator_policy.py bootstrap --account-id $account --region us-east-1 --output $policy
Get-Content -Raw $policy
```

This grants inspection of the account's credits, price catalog queries through the US East (N. Virginia) endpoint, named EC2 inventory and CloudWatch metric reads in the deployment region, and access to the On-Demand G/VT quota and its request history. EC2 inventory, CloudWatch metric reads and billing credit reads have account-level visibility. TokenFlow adds no billing, identity or infrastructure write permissions in this mode. [Billing actions](https://docs.aws.amazon.com/service-authorization/latest/reference/list_billing.html), [Price List endpoints](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-price-list-query-api.html), [CloudWatch actions](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazoncloudwatch.html)

If the operator should submit quota requests, add `--allow-quota-request` when rendering. The resulting permission applies to EC2 quota `L-DB2E81BA` in that account and region. AWS exposes no IAM condition for the requested numeric value. If approval is for exactly **4 total vCPUs**, an administrator can file that request and leave the operator on the read-only bootstrap policy. The account may first need its Service Quotas service-linked role, which an administrator can prepare separately. [Quota resources and condition keys](https://docs.aws.amazon.com/service-authorization/latest/reference/list_service-quotas.html), [Request prerequisites](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html)

## 2. Assign and verify access

1. In the **organization instance of IAM Identity Center**, open **Multi-account permissions → Permission sets → Create permission set → Custom permission set**. Name it `TokenFlowOperator` and paste the rendered JSON under **Inline policy**. Use this as the sole permissions policy for the set. [Create a permission set](https://docs.aws.amazon.com/singlesignon/latest/userguide/howtocreatepermissionset.html)
2. Under **AWS accounts**, select the workload account and choose **Assign users or groups**. Select the operator or operator group and `TokenFlowOperator`. After changing an existing set, provision its updated permissions to the assigned account. [Account assignments](https://docs.aws.amazon.com/singlesignon/latest/userguide/assignusers.html), [Update a permission set](https://docs.aws.amazon.com/singlesignon/latest/userguide/howtoviewandchangepermissionset.html)
3. In PowerShell, sign in through the access portal, select that account and permission set, and check the resulting role and quota.

```powershell
aws configure sso --profile default
aws sso login --profile default
aws sts get-caller-identity --profile default
aws service-quotas get-service-quota --profile default --region us-east-1 --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.{Name:QuotaName,vCPUs:Value}'
```

## 3. Add one worker pool

Use the same PowerShell session and approved values from the deployment catalog:

```powershell
$ami = Read-Host "Approved GPU AMI ID"
$subnet = Read-Host "Approved private subnet ID"
$group = Read-Host "Approved worker security group ID"
$namespace = Read-Host "Exact TokenFlow deployment namespace"
$pool = Read-Host "Exact TokenFlow pool name"
python scripts/aws_operator_policy.py workers --account-id $account --region us-east-1 --namespace $namespace --pool $pool --ami $ami --subnet $subnet --security-group $group --instance-type g4dn.xlarge --output $policy
```

Review and provision the replacement inline policy. Instance launches require the configured type, namespace, pool, name tag, a 32-character resource key and IMDSv2. EC2 separately authorizes the approved AMI, subnet, security groups, private interface and newly allocated EBS disks. Repeat `--security-group` for each approved group. Termination requires the same namespace, pool and key tags. Tag writes apply only during launch. Reserve these tag names for TokenFlow across other roles that can create or change EC2 tags. [EC2 launch authorization](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ExamplePolicies_EC2.html), [Tag-on-create authorization](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/supported-iam-actions-tagging.html)

The worker policy assumes an AMI and model that can run without an instance profile. A profile for gated artifacts requires a separate grant on its specific role. Billing limits, total runtime, aggregate disks and total replica count need separate controls. The guarded live-test script also requires its own cleanup workflow permissions and per-run namespace. Use the dedicated guard role and approval process for that workflow. Keep the generated account-specific JSON outside source control.
