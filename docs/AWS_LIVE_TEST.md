# One private GPU validation in us-east-1

This test exercises one on-demand `g4dn.xlarge` (one T4) with Qwen2.5-0.5B-Instruct
in FP16. It captures a JSON completion, a complete SSE stream, engine gauges and
physical GPU utilization. An in-process gateway uses the regular controller to
discover the tagged private worker. The test submits one allocation with EC2's
idempotent client token and prohibits the adoption controller from launching
replacements. This checks serving and discovery. It does not benchmark
throughput, measure streaming latency, or deploy the public gateway.

## Existing infrastructure

Use a control host with the project's AWS Python extra and an existing private
route to the worker subnet in `us-east-1`. This can be a host in another subnet
or a client on a VPN. Verify that this path and its return route carry TCP
8000/8001 before setting `--confirm-private-route`. The dedicated worker
security group must allow those ports from the control host's group or trusted
private source CIDR only. On a VPN, use `--source-cidr` with the CIDR that the
worker actually sees. Check route tables, security groups and network ACLs.
The command creates no networking or public endpoint.

For an AMI with Qwen2.5-0.5B-Instruct and the image already loaded, pass
`--local-checkpoint /path/on/ami` and optionally `--image your-local-oci-tag`.
The test uses a read-only checkpoint mount and Docker's never-pull policy.
This mode works in an isolated subnet. Otherwise verify outbound HTTPS and
DNS for Docker Hub and Hugging Face, and check that artifact transfer fits the
$2 allowance before setting `--confirm-artifact-egress`.

Choose a GPU-ready x86_64 Linux AMI without software licensing charges.
It needs Docker, a compatible NVIDIA driver/container toolkit, room for the
vLLM image and checkpoint, and baseline gp3 volumes with DeleteOnTermination.
The test uses the AMI's existing disk sizes. Use an account where the applied
on-demand G/VT quota has at least four available vCPUs. Capacity reservations
also consume quota. [EC2 instance quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html)

## Price and credit check

Run the following in the checkout after setting the three infrastructure IDs
and the control group's ID. Without `--execute`, this command is read-only.

```bash
python scripts/aws_live_test.py --ami "$AMI" --subnet "$SUBNET" \
  --worker-group "$WORKER_SG" --source-group "$CONTROL_SG"
```

It queries the regional Linux shared-tenancy GPU price and gp3 storage price,
the account's G/VT quota, its current commitments and its eligible promotional
balance through AWS Billing. It counts same-account, enabled USD promotion
credits valid for EC2 over the test window. It uses the lower of estimated and
settled remaining amounts when AWS reports both. Use `--credit-description`
to select a specific award. Missing permissions, capacity or credits stop the
run. Use a current boto3 release that includes this Billing API. The estimate includes two
hours of EC2 and EBS plus a $2 allowance for transfer and workflow overhead.
It refuses estimates above $10 and prints the STS account ID. The deadline is
30 minutes. The estimate is not an AWS billing limit. Existing infrastructure,
actual transfer charges, tax and the applicability of promotional credits depend
on the account. A root session can inspect the plan. Execution requires a
non-root principal. [AWS Billing GetCredits](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_billing_GetCredits.html)

## Reusable cleanup guard

Once per test account, the operator installs the supplied Step Functions
Standard workflow and execution role. These are persistent operator
infrastructure shared across test runs. Install only after approving the
estimate and confirming credits. Do not remove or alter the guard while an
execution is active.

```bash
aws cloudformation deploy --region us-east-1 --stack-name tokenflow-gpu-test-guard \
  --template-file deploy/aws-test-guard.yaml --capabilities CAPABILITY_IAM
```

The execution name is a fresh random test namespace. Before requesting a GPU,
the launcher checks the stack template, simulates the guard role's tagged
termination permission, and verifies that the execution successfully queried
EC2 and entered its deadline wait. At the deadline the workflow repeatedly
discovers this test's tagged instances and requests termination. It checks for
late inventory through ten minutes after the deadline and continues polling
while tagged instances remain visible. This runs in AWS if the control host or
client disappears. IAM policy simulation cannot rule out every runtime denial.
[AWS SDK workflow integrations](https://docs.aws.amazon.com/step-functions/latest/dg/supported-services-awssdk.html)

## Run and inspect

Set `ACCOUNT_ID` to the 12-digit account that owns the approved credit. The launcher checks it against STS before querying billing or EC2, then repeats its price, quota, network and credit checks on execution.
This example uses the fixed public image and checkpoint.

```bash
python scripts/aws_live_test.py --ami "$AMI" --subnet "$SUBNET" \
  --worker-group "$WORKER_SG" --source-group "$CONTROL_SG" \
  --expected-account "$ACCOUNT_ID" --confirm-private-route --confirm-artifact-egress --execute
```

The caller needs read access to STS, AWS Billing credits, EC2 inventory and network configuration,
Service Quotas, Price List, CloudFormation and IAM policy simulation, plus
execution/history/stop access to the guard. It needs the regular launch/tag
and scoped termination permissions of the cloud provider. The control host
does not pass a worker instance profile. The guard role can terminate only
instances tagged for live tests.

Look in `experiments/results/aws-live/tf-test-*/` for `run.json` (account,
estimate, execution ARN and deadline), the local catalog and state database,
and `responses.json` (successful replies and sampled worker telemetry).
The client confirms instance termination and disappearance of the observed
volume and network interface IDs before disarming the independent guard.
After a disconnect or ambiguous launch, check the Step Functions execution
and EC2 resources by `TokenFlowNamespace`. A failed workflow or lingering
resource requires operator intervention. The workflow execution eventually
expires from AWS history. To uninstall the reusable guard when no test is
active, delete its CloudFormation stack.
