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

If you normally choose an account and role in an IAM Identity Center portal, use this flow in place of `aws login`. Have your organization's start URL and SSO region ready. The SSO region can differ from the EC2 region.

```powershell
& $Aws configure sso --profile default
& $Aws sso login --profile default
& $Aws configure set region us-east-1 --profile default
& $Aws sts get-caller-identity --profile default
```

Install TokenFlow with its `aws` extra in the Python environment that will run the gateway. This includes the SDK support for cached login credentials. Authentication can be verified without launching an instance. [Boto3 credential providers](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)
