# TokenFlow

TokenFlow is a distributed LLM inference gateway and control plane. The current
phase provides a FastAPI service and an in-memory registry for future inference
workers.

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

## Test

```powershell
pytest
```

The registry is intentionally process-local and in-memory. Registered workers
are lost when the gateway restarts.
