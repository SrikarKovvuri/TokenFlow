"""Shared container entry point and validated operator-supplied engine settings."""

from pathlib import Path, PurePosixPath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tokenflow.provisioner.base import PoolSpec


class EngineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_model_len: int | None = Field(default=None, ge=1)


def check_mount_path(value: str) -> str:
    if not PurePosixPath(value).is_absolute() or any(c in value for c in ",:\x00\r\n"):
        raise ValueError("Mount paths must be absolute Linux paths without commas, colons or newlines")
    return value


def local_checkpoint_paths(checkpoint: str) -> tuple[str, str]:
    """Return the read-only host bind source and worker-visible checkpoint."""
    path = PurePosixPath(checkpoint)
    if ".." not in path.parts and path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", path.name):
        repository = path.parent.parent
        kind, *parts = repository.name.split("--")
        if kind == "models" and len(parts) in (1, 2) and all(
            re.fullmatch(r"[\w](?:[\w.-]*[\w])?", part, re.ASCII) and ".." not in part for part in parts
        ):
            # Snapshot files link to ../../blobs inside this model repository.
            return str(repository), f"/checkpoint/snapshots/{path.name}"
    return checkpoint, "/checkpoint"


def worker_script() -> str:
    return Path(__file__).with_name("worker.py").read_text(encoding="utf-8")


def worker_args(pool: PoolSpec, options: EngineOptions, port: int, checkpoint: str) -> list[str]:
    args = ["--backend", pool.backend, "--model", pool.model, "--checkpoint", checkpoint,
            "--gpus", str(pool.gpu_count), "--gpu-type", pool.gpu_type, "--capacity",
            str(pool.capacity), "--port", str(port), "--dtype", options.dtype]
    if options.max_model_len:
        args.extend(["--max-model-len", str(options.max_model_len)])
    return args
