# Operating a managed gateway

Students need an HTTPS base URL, an inference key, and a model name. The operator owns the provider account, the catalog of permitted checkpoints and hardware, and the maximum number of allocations. Calls that specify a model outside that catalog cannot create new pools or machines.

## Choose a control host

Run one gateway process on an always-on Linux service host with durable local storage. Slurm pools require working Slurm clients, a shared project directory, and network access to the compute nodes. On Insomnia, use a service host approved for long-running processes. The login node is for short submissions and queries. On AWS, place the control host in a VPC with private routes to GPU instances. A gateway combining both providers needs both sets of routes and credentials.

Keep a stable DNS name on the gateway. When Slurm reassigns a node or AWS replaces an instance, the controller updates the private worker address. The student's base URL stays the same. Use a trusted network between gateway and workers for their HTTP traffic.

## Configure a deployment

1. Install Python 3.11+ and check out the repository on the service host. Create the configuration directory with `sudo install -d -m 0750 /etc/tokenflow`. The service account must have read access to `platform.json` and permission to traverse its parent directory.
2. Copy the matching example from `deploy/` to `/etc/tokenflow/platform.json`. Give the deployment a unique namespace. Use a local, durable `state_path` and reuse it across restarts. One deployment owns one namespace and one database.
3. Replace the example paths or AWS IDs. On Insomnia, use the absolute SIF and checkpoint paths on shared storage, and an owner-only shared project directory for `options.state_dir`. Set the allocation account, partition, QoS and GPU resource spelling to those available to the service account. The example uses the A6000. A separate pool can request H100 or L40S with its own capacity and checkpoint. For AWS, supply the GPU-ready AMI, private subnet, security groups and IAM permissions described in the provider guide. The example IDs are placeholders.
4. Set `capacity` to the maximum number of concurrent requests per worker, and `service_time_seconds` to an initial service-time estimate for this model/hardware. Give interchangeable workers the same served model alias. Use distinct aliases for checkpoints with different semantics. Set `min_replicas` for warm capacity and `max_replicas` as a hard per-pool allocation ceiling. Each replica uses `gpu_count` GPUs.

From the checkout on the **Linux control host**:

```bash
sudo python3 -m venv /opt/tokenflow/venv
sudo /opt/tokenflow/venv/bin/pip install '.[aws]'
/opt/tokenflow/venv/bin/tokenflow validate /etc/tokenflow/platform.json
```

Validation reads the configuration offline. GPU availability, IAM permissions, cluster quotas, driver compatibility and model memory still require live verification. See [provider setup](PROVIDERS.md), [AWS sign-in from Windows](AWS_SIGN_IN.md), and [the bounded single-GPU validation](AWS_LIVE_TEST.md).

## Start a persistent HTTPS endpoint

On a new deployment create a private environment file. The systemd unit reads it on startup. Keep the worker secret stable across gateway restarts. Give each student an individual inference key in the JSON array. Reserve the administrative key for the operator.

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

Install `deploy/tokenflow.service` as `/etc/systemd/system/tokenflow.service`. Set its `User` and `Group` to the service account that can submit the jobs or assume the AWS role. The example expects an existing `tokenflow` Unix account. For this account, make `/etc/tokenflow` traversable with `sudo chgrp tokenflow /etc/tokenflow`, and make the catalog readable with `sudo chown root:tokenflow /etc/tokenflow/platform.json` and `sudo chmod 0640 /etc/tokenflow/platform.json`. Keep `keys.env` owned by root and mode 0600. Systemd reads the environment file and creates the durable state directory.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tokenflow
curl --fail http://127.0.0.1:8000/healthz
sudo journalctl -u tokenflow -n 50
```

Put a TLS reverse proxy in front of the loopback service. For Caddy, install `deploy/Caddyfile` and set `TOKENFLOW_HOSTNAME` in **Caddy's** service environment to the assigned DNS name. Route that name to the service host. Use the campus TLS ingress if the service is reachable only through the campus network. Public ACME validation with the default Caddy configuration requires DNS and inbound certificate-validation connectivity. [Caddy HTTPS setup](https://caddyserver.com/docs/quick-starts/reverse-proxy#https)

## Operate and verify

Use a private operator terminal with `BASE_URL` set to the gateway origin and `ADMIN_KEY` set to the administrative key. These commands also work through a trusted loopback connection. Each command targets the example Slurm pool.

```bash
curl --fail -H "Authorization: Bearer $ADMIN_KEY" "$BASE_URL/v1/pools"
curl --fail -H "Authorization: Bearer $ADMIN_KEY" "$BASE_URL/v1/workers"
curl --fail -X PUT -H "Authorization: Bearer $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"minimum":1,"maximum":1}' "$BASE_URL/v1/pools/qwen-a6000/limits"
```

Watch the pool for a provider job or instance ID, then check that its registered worker becomes healthy and reports telemetry. Run one streaming chat request using the student example in the README. For a cold pool, the first request records demand and can return HTTP 503 while resources start. A busy pool at capacity returns HTTP 429. A warm minimum of one avoids the initial allocation delay.

To pause a pool, repeat the PUT with `{"minimum":0,"maximum":0}`. TokenFlow stops assigning new requests to retiring workers, lets their active requests finish, and releases their allocations. Wait for an empty resource list and check the provider before tearing down the control host. Stopping the gateway by itself leaves the provider allocations in place for recovery. SQLite stores pool overrides and allocation identities. Keep the database on a durable local volume, back it up, and keep a single active gateway per namespace.

For troubleshooting, compare `last_error` and resource phases in `/v1/pools` with Slurm jobs or the EC2 instance state. Check the worker startup log for image, GPU or checkpoint failures. Engine metrics and physical GPU utilization are independent signals. Missing measurements remain unknown. Read [scheduling and admission](SCHEDULING.md) for how the gateway uses them.
