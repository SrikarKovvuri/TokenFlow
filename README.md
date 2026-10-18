# TokenFlow

TokenFlow is a distributed LLM inference gateway and control plane. It
currently provides an in-memory worker registry, periodic health checks, and
single-worker forwarding for OpenAI-compatible chat completions.

## Requirements

- Python 3.11 or newer

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run

```powershell
uvicorn tokenflow.main:app --reload
```

The API is available at `http://localhost:8000`, with interactive documentation
at `http://localhost:8000/docs`.

## Connect an inference worker

Start a vLLM server separately, then register its reachable base URL through
`POST /v1/workers`. TokenFlow does not currently submit Slurm jobs, allocate
GPUs, or start vLLM processes.

New workers begin with `unknown` health. TokenFlow periodically requests the
worker's `/health` endpoint and changes its state to `healthy` or `unhealthy`.
Only enabled, healthy workers can receive requests through:

```text
POST /v1/chat/completions
```

Both regular JSON responses and SSE streams are relayed. No Columbia worker
address is stored in this repository; register the address at runtime once it
is available.

Relevant environment settings include:

```text
TOKENFLOW_WORKER_HEALTH_INTERVAL_SECONDS
TOKENFLOW_WORKER_HEALTH_TIMEOUT_SECONDS
TOKENFLOW_WORKER_CONNECT_TIMEOUT_SECONDS
TOKENFLOW_WORKER_REQUEST_TIMEOUT_SECONDS
```

## Test

```powershell
pytest
```

The registry is intentionally process-local and in-memory. Registered workers
are lost when the gateway restarts.
