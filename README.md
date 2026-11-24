# TokenFlow

TokenFlow gives inference clients one OpenAI-compatible endpoint for models running on Columbia Slurm workers or AWS EC2 GPU instances. The gateway serves the model aliases configured by the operator. A shared controller allocates workers, discovers their addresses, and drains idle capacity. vLLM or SGLang executes each request.

## Use an existing deployment

Ask the operator for the **base URL ending in `/v1`**, an inference key, and the model alias. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in the client environment. With the OpenAI Python client installed:

```python
from openai import OpenAI

client = OpenAI()
print([model.id for model in client.models.list().data])
stream = client.chat.completions.create(
    model="qwen-14b",
    messages=[{"role": "user", "content": "What is tensor parallelism?"}],
    stream=True,
)
for event in stream:
    if event.choices:
        print(event.choices[0].delta.content or "", end="", flush=True)
```

Omit `stream=True` for a single JSON completion. Students use the same endpoint when GPU jobs move or AWS instances change. A cold model can return HTTP 503 during startup. A saturated pool returns HTTP 429 with a retry hint.

## Run the platform

- **Columbia HPC.** Run the gateway on a service host with Slurm access, shared project storage, and a route to compute nodes. TokenFlow submits a GPU job, launches the Apptainer image, and discovers the node and port.
- **AWS.** Run the gateway on a persistent service host with an EC2 role and private routes to workers. TokenFlow creates tagged GPU instances, launches the configured engine in Docker, and discovers each private address.
- **Local development.** Run the gateway on your laptop or development VM. Register an existing OpenAI-compatible worker with `POST /v1/workers`.

For a managed deployment, start with [the operator guide](docs/PLATFORM.md). It covers credentials, a durable SQLite state volume, HTTPS, service installation, warm and cold pools, and shutting down GPU spend. The [provider guide](docs/PROVIDERS.md) describes hardware, checkpoint, network and permission requirements. The templates in [deploy](deploy/) start with zero warm replicas.

## How a request gets served

1. The gateway authenticates the client and reads the requested model. Only enabled, healthy workers serving that alias are eligible.
2. The scheduler reserves a slot using worker capacity, estimated service time, and recent engine load. KV-cache occupancy and per-device GPU utilization help choose among workers with equal estimated delay.
3. The gateway forwards the completion and holds the slot until the response or SSE stream closes. A connection failure can be retried on another eligible worker before inference starts.
4. The controller reconciles recent demand with each pool's replica limits. It adopts existing allocations after a gateway restart, registers new endpoints, and waits for outstanding requests to finish before terminating a draining worker.

The monitoring signals have timestamps and fallbacks. Service-time estimates need calibration with the workloads you serve. See [scheduling and admission](docs/SCHEDULING.md).

## Local development

Requires Python 3.11 or newer. From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
uvicorn tokenflow.main:app --reload --reload-dir src
```

The development gateway listens on loopback at `http://127.0.0.1:8000`. The interactive API reference is at `/docs`. With no platform configuration it uses an in-memory registry. Keep this unauthenticated development mode on loopback. Managed mode requires separate inference and admin keys.

The gateway exposes `GET /healthz`, `GET /v1/models`, and `POST /v1/chat/completions`. Operators use `/v1/workers` for endpoint health, active requests and telemetry, `GET /v1/pools` for provider allocations, and `PUT /v1/pools/{name}/limits` for persistent scaling limits. Run exactly one gateway process per deployment.

## Experiments

The [experiment suite](experiments/README.md) records streaming timings, throughput and engine telemetry. For the original local gateway and SSH tunnel workflow, `python experiments/smoke.py` discovers the served model, registers the worker and verifies JSON and SSE. Historical cluster runbooks and benchmark results describe those individual runs.
