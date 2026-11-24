# Operating a managed gateway

Students need an HTTPS base URL, an inference key, and a model name. The operator owns the provider account, the catalog of permitted checkpoints and hardware, and the maximum number of allocations. Calls that specify a model outside that catalog cannot create new pools or machines.

## Choose a control host

Run one gateway process on an always-on Linux service host with durable local storage. Slurm pools require working Slurm clients, a shared project directory, and network access to the compute nodes. On Insomnia, use a service host approved for long-running processes. The login node is for short submissions and queries. On AWS, place the control host in a VPC with private routes to GPU instances. A gateway combining both providers needs both sets of routes and credentials.

Keep a stable DNS name on the gateway. When Slurm reassigns a node or AWS replaces an instance, the controller updates the private worker address. The student's base URL stays the same. Use a trusted network between gateway and workers for their HTTP traffic.

## Configure a deployment

Install Python 3.11+, pip and venv. Run the commands below from the checkout in
one Bash session on the Linux control host. For Insomnia, set `svc_user` to the
existing Unix account authorized to submit Slurm jobs and read the shared SIF
and checkpoints. For AWS, use a dedicated unprivileged Unix account and attach
the control-plane IAM role to its host. On a fresh AWS control host:

```bash
svc_user=tokenflow
if ! id "$svc_user" >/dev/null 2>&1; then
  sudo useradd --system --user-group --no-create-home --home-dir /var/lib/tokenflow \
    --shell /usr/sbin/nologin "$svc_user"
fi
```

On a fresh deployment, set `example` to `aws` or `insomnia`. Install the catalog
under the service user's primary group before running validation. Keep the
existing catalog when upgrading a deployment.

```bash
svc_group=$(id -gn "${svc_user:?set the service account first}")
example=aws
sudo install -d -o root -g "$svc_group" -m 0750 /etc/tokenflow
sudo install -o root -g "$svc_group" -m 0640 "deploy/$example.example.json" /etc/tokenflow/platform.json
sudoedit /etc/tokenflow/platform.json
```

Give each deployment a unique namespace and a durable local `state_path`. On
Insomnia, enter the local Slurm account, partition, QoS and GPU resource
spelling. Choose an owner-only shared project directory for `options.state_dir`.
On AWS, supply the actual AMI, subnet and security groups. Set `capacity` and
`service_time_seconds` for each model/hardware combination. Use `min_replicas`
for warm capacity and `max_replicas` to cap allocations.

Build the wheel and its dependencies as the ordinary operator. Install the
built wheels in a root-owned environment, then validate the exact catalog and
executable as the service identity.

```bash
wheel_dir=$(mktemp -d)
python3 -m pip wheel --wheel-dir "$wheel_dir" '.[aws]'
sudo python3 -m venv /opt/tokenflow/venv
sudo /opt/tokenflow/venv/bin/pip install --no-index --find-links "$wheel_dir" 'tokenflow[aws]'
sudo -u "$svc_user" /opt/tokenflow/venv/bin/tokenflow validate /etc/tokenflow/platform.json
```

Validation reads the configuration offline. GPU availability, IAM permissions, cluster quotas, driver compatibility and model memory still require live verification. See [provider setup](PROVIDERS.md), [AWS sign-in from Windows](AWS_SIGN_IN.md), and [the bounded single-GPU validation](AWS_LIVE_TEST.md).

## Start a persistent HTTPS endpoint

On a new deployment create a private environment file. The systemd unit reads it on startup. Keep the worker secret stable across gateway restarts. Give each student their individual inference key through the approved credential-sharing channel. The example generates one student key. Reserve the administrative key for the operator.

```bash
sudo python3 - <<'PY'
import json, os, secrets
from pathlib import Path
os.umask(0o077)
with Path("/etc/tokenflow/keys.env").open("x") as f:
    print("TOKENFLOW_ADMIN_KEY=" + secrets.token_urlsafe(32), file=f)
    print("TOKENFLOW_WORKER_SECRET=" + secrets.token_urlsafe(32), file=f)
    print("TOKENFLOW_API_KEYS='" + json.dumps([secrets.token_urlsafe(32)]) + "'", file=f)
PY
```

Keep `keys.env` owned by root and mode 0600. Systemd reads that file and
creates the durable state directory. Install the unit and set its runtime
identity to the same account that passed catalog validation.

```bash
sudo install -o root -g root -m 0644 deploy/tokenflow.service /etc/systemd/system/tokenflow.service
sudo install -d -m 0755 /etc/systemd/system/tokenflow.service.d
printf '[Service]\nUser=%s\nGroup=%s\n' "$svc_user" "$svc_group" | \
  sudo tee /etc/systemd/system/tokenflow.service.d/identity.conf >/dev/null
sudo systemd-analyze verify /etc/systemd/system/tokenflow.service
sudo systemctl daemon-reload
sudo systemctl enable --now tokenflow
curl --fail http://127.0.0.1:8000/healthz
sudo journalctl -u tokenflow -n 50
```

Put a TLS reverse proxy in front of the loopback service. For a dedicated
host with the packaged Caddy system service installed, set `TOKENFLOW_HOSTNAME`
in this shell to the assigned DNS name. Route that name to this host. For an
existing multisite Caddy installation, merge the supplied site block into the
existing configuration. Public certificate issuance needs working DNS and
inbound ACME validation. Campus deployments can use the approved TLS ingress.
[Caddy HTTPS setup](https://caddyserver.com/docs/quick-starts/reverse-proxy#https)

```bash
: "${TOKENFLOW_HOSTNAME:?set your assigned gateway hostname}"
sudo install -o root -g root -m 0644 deploy/Caddyfile /etc/caddy/Caddyfile
sudo install -d -m 0755 /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironment=TOKENFLOW_HOSTNAME=%s\n' "$TOKENFLOW_HOSTNAME" | \
  sudo tee /etc/systemd/system/caddy.service.d/tokenflow.conf >/dev/null
sudo -H -u caddy env TOKENFLOW_HOSTNAME="$TOKENFLOW_HOSTNAME" \
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl daemon-reload
sudo systemctl enable caddy
sudo systemctl restart caddy
BASE_URL="https://$TOKENFLOW_HOSTNAME"
curl --fail "$BASE_URL/healthz"
```

Caddy flushes `text/event-stream` responses immediately by default. Once a
worker is healthy, run the streaming student example in the README against
`$BASE_URL/v1`. Tokens should arrive before the request finishes. Interrupt a
longer request and check that the worker's `active_requests` returns to zero.
[Caddy streaming behavior](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming)

To check the proxy before deployment, install the development dependencies
and set `TOKENFLOW_TEST_CADDY` to the absolute path of a Caddy 2 executable.
With loopback port 8000 free, run
`python -m pytest tests/test_openai_client.py -q`. The HTTPS case imports the
shipped site, uses a temporary local CA, and lets the mock worker send its
second chunk only after the OpenAI client receives the first.

## Operate and verify

Use a private Bash terminal with `BASE_URL` set to the gateway origin. Enter
the administrative key at the hidden prompt. Curl reads it from standard input
to keep it out of its process arguments. These examples target the Slurm pool.

```bash
read -r -s -p 'TokenFlow admin key: ' ADMIN_KEY
echo
curl --fail -H @- "$BASE_URL/v1/pools" <<<"Authorization: Bearer $ADMIN_KEY"
curl --fail -H @- "$BASE_URL/v1/workers" <<<"Authorization: Bearer $ADMIN_KEY"
curl --fail -X PUT -H @- -H 'Content-Type: application/json' \
  -d '{"minimum":1,"maximum":1}' "$BASE_URL/v1/pools/qwen-a6000/limits" <<<"Authorization: Bearer $ADMIN_KEY"
unset ADMIN_KEY
```

Watch the pool for a provider job or instance ID, then check that its registered worker becomes healthy and reports telemetry. Run one streaming chat request using the student example in the README. For a cold pool, the first request records demand and can return HTTP 503 while resources start. A busy pool at capacity returns HTTP 429. A warm minimum of one avoids the initial allocation delay.

To pause a pool, repeat the PUT with `{"minimum":0,"maximum":0}`. TokenFlow stops assigning new requests to retiring workers, lets their active requests finish, and releases their allocations. Wait for an empty resource list and check the provider before tearing down the control host. Stopping the gateway by itself leaves the provider allocations in place for recovery. SQLite stores pool overrides and allocation identities. Keep the database on a durable local volume, back it up, and keep a single active gateway per namespace.

After a gateway restart, each rediscovered managed worker keeps its worker ID. Its health, telemetry and in-flight request count are measured again in the new process. A worker can serve requests once it passes a fresh health check. For EC2, a resource marked `terminating` still occupies one replica slot while AWS shuts it down. The slot becomes available when discovery reports termination or the instance stays absent through the inventory grace period.

For troubleshooting, compare `last_error` and resource phases in `/v1/pools` with Slurm jobs or the EC2 instance state. Check the worker startup log for image, GPU or checkpoint failures. Engine metrics and physical GPU utilization are independent signals. Missing measurements remain unknown. Read [scheduling and admission](SCHEDULING.md) for how the gateway uses them.
