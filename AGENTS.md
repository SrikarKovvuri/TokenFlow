# Working on TokenFlow

TokenFlow coordinates independent inference engines. Keep model execution in vLLM or SGLang and provider-specific resource management behind the provisioner boundary. Preserve the local/manual gateway used by the existing experiments.

## Code map

- `src/tokenflow/main.py` wires HTTP clients, authentication, health polling and the controller into one FastAPI lifespan.
- `worker_manager/` owns worker records, enabled/health state, reservations and observations. Check the probed URL before applying a health or metrics sample. Keep assignment and reservation atomic within the gateway event loop.
- `scheduler/` selects eligible workers, applies admission control and relays JSON/SSE responses. Release reservations on every exit path, including downstream disconnects. An ambiguous upstream result cannot be safely replayed.
- `platform/` loads the operator's pool catalog, stores desired capacity and allocation identities, and reconciles provider state. Preserve draining workers until their in-flight requests complete. Discovery failure and an empty successful inventory have different meanings.
- `provisioner/` defines the common provider contract, Slurm/Apptainer and EC2 adapters, and the container entry point. Keep blocking provider SDK or CLI calls off the HTTP event loop. Scope discovery and termination to this deployment and pool.
- `experiments/` and `scripts/` contain benchmarking utilities and historical workflows. Write per-run results to a dedicated output directory.

## Interfaces and verification

Support Python 3.11+ and one active gateway process per deployment. Manual worker registrations are process-local. Managed pool limits and allocation identities survive a process restart. Use absolute operator-supplied paths and private worker addresses discovered at runtime. Keep credentials, cluster node addresses and AWS account identifiers out of source control.

Run `python -m pytest -q` from the repository root. Provider tests should use fake inventories, subprocess results or SDK stubs. Describe any live validation separately, including the actual engine, GPU, and whether allocations were released. Keep platform and provider configuration examples in sync with their validators.

See [the operator guide](docs/PLATFORM.md), [provider guide](docs/PROVIDERS.md) and [scheduler guide](docs/SCHEDULING.md).
