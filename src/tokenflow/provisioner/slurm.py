"""Run single-node Apptainer workers through Slurm's native client commands."""

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints

from tokenflow.provisioner.base import PoolSpec, Resource
from tokenflow.provisioner.launch import EngineOptions, check_mount_path, worker_args, worker_script


SlurmName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_.-]+$")]
KEY = re.compile(r"[0-9a-f]{32}")


class SlurmOptions(EngineOptions):
    account: SlurmName
    partition: SlurmName
    qos: SlurmName | None = None
    cpus: int = Field(default=8, ge=1)
    memory: Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*[KMGT]?$")] = "32G"
    time_limit: Annotated[str, StringConstraints(pattern=r"^(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$")] = "02:00:00"
    state_dir: Annotated[str, AfterValidator(check_mount_path)]
    apptainer_module: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9._/-]+$")] | None = None


def worker_token(secret: str, key: str) -> str:
    return hmac.new(secret.encode(), key.encode(), hashlib.sha256).hexdigest()


class SubmissionRejected(RuntimeError):
    """Slurm explicitly rejected a submission before allocating a job ID."""


def slurm_command(*args: str, script: str | None = None) -> str:
    result = subprocess.run(args, input=script, text=True, capture_output=True, timeout=30)
    if result.returncode != 0:
        detail = result.stderr.strip()
        # A failed process alone is insufficient: the RPC reply could have been
        # lost after acceptance. Only recognized pre-submit errors release a key.
        rejected = re.search(
            r"Batch job submission failed: (?:Invalid (?:account|partition|qos|generic resource)|"
            r"Requested (?:node configuration|time limit) is not available|"
            r"Job violates accounting/QOS policy|Access/permission denied)|"
            r"sbatch: (?:unrecognized option|error: invalid (?:partition|account|time) )",
            detail, re.IGNORECASE,
        )
        if args[0] == "sbatch" and rejected and not result.stdout.strip():
            raise SubmissionRejected(detail)
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {detail}")
    return result.stdout


def private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    stat = path.lstat()
    if path.is_symlink() or not path.is_dir() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError(f"Slurm state must be a private directory (0700) owned by this user: {path}")


def private_file(path: Path, value: str) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        output.write(value)


def batch_script(pool: PoolSpec, opts: SlurmOptions, state: Path) -> str:
    args = worker_args(pool, opts, 0, "/checkpoint")
    module = f"module load {shlex.quote(opts.apptainer_module)}" if opts.apptainer_module else ""
    # Pass only this job's derived credential on the private shared mount. Slurm
    # can expose submitted scripts to administrators, so scripts contain no keys.
    return f"""#!/bin/bash -l
set -euo pipefail
umask 077
{module}
work=$(mktemp -d "${{TMPDIR:-/tmp}}/tokenflow-worker.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
export APPTAINER_TMPDIR="$work/image"
mkdir -p "$APPTAINER_TMPDIR" "$work/runtime/cache"
cat > "$work/runtime/worker.py" <<'TOKENFLOW_PY'
{worker_script()}
TOKENFLOW_PY
export APPTAINERENV_XDG_CACHE_HOME=/work/cache APPTAINERENV_HF_HOME=/work/cache/huggingface
export APPTAINERENV_VLLM_CACHE_ROOT=/work/cache/vllm APPTAINERENV_TRITON_CACHE_DIR=/work/cache/triton
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor APPTAINERENV_CUDA_CACHE_PATH=/work/cache/cuda
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated devices}}"
apptainer exec --cleanenv --nv --pwd /work --bind "$work/runtime:/work" \\
  --bind {shlex.quote(str(state) + ":/state")} --bind {shlex.quote(pool.checkpoint + ":/checkpoint:ro")} \\
  {shlex.quote(pool.image)} python3 /work/worker.py {shlex.join(args)} \\
  --key-file /state/key --port-file "/state/port-$SLURM_JOB_ID"
"""


class SlurmProvisioner:
    def __init__(self, namespace: str, worker_secret: str):
        if not worker_secret:
            raise ValueError("A worker secret is required for Slurm")
        self.namespace = namespace
        self.worker_secret = worker_secret

    @staticmethod
    def validate_pool(pool: PoolSpec) -> None:
        SlurmOptions.model_validate(pool.options)
        check_mount_path(pool.image)
        check_mount_path(pool.checkpoint)

    def _prefix(self, pool: PoolSpec) -> str:
        digest = hashlib.sha256(f"{self.namespace}/{pool.name}".encode()).hexdigest()[:24]
        return f"tf-{digest}-"

    def _discover(self, pool: PoolSpec) -> list[Resource]:
        state_dir = Path(SlurmOptions.model_validate(pool.options).state_dir)
        prefix = self._prefix(pool)
        rows = slurm_command("squeue", "--local", "--me", "--noheader", "--states=all", "--format=%i|%j|%T|%B")
        resources = []
        for row in rows.splitlines():
            fields = row.strip().split("|")
            if len(fields) != 4:
                continue
            job_id, name, state, host = fields
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):]
            if not KEY.fullmatch(key) or not job_id.isdecimal():
                continue
            resource = Resource(key=key, external_id=job_id, phase="pending")
            if state in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
                         "OUT_OF_MEMORY", "DEADLINE", "BOOT_FAIL", "PREEMPTED", "REVOKED"}:
                resource = Resource(key=key, external_id=job_id, phase="terminal")
            elif state in {"CONFIGURING", "COMPLETING", "STOPPED"}:
                # Slurm still has an allocation. Hold its slot without routing
                # while nodes boot, stop, or finish their epilog and release GPUs.
                resource = Resource(key=key, external_id=job_id, phase="running")
            elif state == "RUNNING":
                # The GPU is allocated and its startup clock has begun even when
                # the engine has not published a usable address yet.
                resource = Resource(key=key, external_id=job_id, phase="running")
                port_file = state_dir / name / f"port-{job_id}"
                try:
                    port = int(port_file.read_text())
                except (FileNotFoundError, ValueError):
                    pass
                else:
                    if 1024 <= port <= 65534 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
                        resource = Resource(key=key, external_id=job_id, phase="running",
                                            base_url=f"http://{host}:{port}")
            resources.append(resource)
        return resources

    async def discover(self, pool: PoolSpec) -> list[Resource]:
        return await asyncio.to_thread(self._discover, pool)

    def _launch(self, pool: PoolSpec, key: str) -> Resource:
        self.validate_pool(pool)
        if not KEY.fullmatch(key):
            raise ValueError("Invalid launch key")
        opts = SlurmOptions.model_validate(pool.options)
        root = Path(opts.state_dir)
        private_directory(root)
        name = self._prefix(pool) + key
        state = root / name
        private_directory(state)
        claim = state / "submitting"
        try:
            private_file(claim, "")
        except FileExistsError:
            # sbatch has no client token. A persisted intent plus a missing reply
            # is ambiguous: issuing another sbatch here could allocate twice.
            for existing in self._discover(pool):
                if existing.key == key:
                    return existing
            receipt = state / "job-id"
            if receipt.exists():
                return Resource(key=key, external_id=receipt.read_text().strip(), phase="pending")
            raise RuntimeError(f"Ambiguous Slurm submission. Inspect {state} and queue before retrying.")

        args = ["sbatch", "--parsable", "--no-requeue", "--export=NIL", f"--job-name={name}",
                "--nodes=1", "--ntasks=1", f"--account={opts.account}", f"--partition={opts.partition}",
                f"--gres=gpu:{pool.gpu_type}:{pool.gpu_count}", f"--cpus-per-task={opts.cpus}",
                f"--mem={opts.memory}", f"--time={opts.time_limit}", f"--chdir={state}",
                f"--output={state}/slurm-%j.out"]
        if opts.qos:
            args.append(f"--qos={opts.qos}")
        try:
            token = worker_token(self.worker_secret, key)
            if not (state / "key").exists():
                private_file(state / "key", token)
            if pool.backend == "sglang" and not (state / "sglang.json").exists():
                private_file(state / "sglang.json", json.dumps({"api-key": token}))
            body = batch_script(pool, opts, state)
        except Exception:
            claim.unlink()  # No Slurm submission has been attempted.
            raise
        try:
            reply = slurm_command(*args, script=body).strip()
        except (SubmissionRejected, FileNotFoundError):
            claim.unlink()
            raise
        job_id = reply.split(";", 1)[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"Unexpected sbatch response for {name}: {reply!r}")
        private_file(state / "job-id", job_id)
        return Resource(key=key, external_id=job_id, phase="pending")

    async def launch(self, pool: PoolSpec, key: str) -> Resource:
        return await asyncio.to_thread(self._launch, pool, key)

    async def terminate(self, pool: PoolSpec, resource: Resource) -> None:
        # A stale externally supplied job ID must never cancel another user's job.
        for job in await self.discover(pool):
            if job.key == resource.key and job.external_id == resource.external_id and job.phase != "terminal":
                await asyncio.to_thread(slurm_command, "scancel", "--name", self._prefix(pool) + job.key, job.external_id)
                return
