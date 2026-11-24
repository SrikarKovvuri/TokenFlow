"""Operator configuration. Loading and validating a file never provisions a GPU."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tokenflow.provisioner.base import Identifier, PoolSpec, Provisioner


class PlatformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    namespace: Identifier
    state_path: Path
    pools: list[PoolSpec] = Field(min_length=1)
    interval_seconds: float = Field(default=10, gt=0)
    idle_seconds: float = Field(default=300, gt=0)
    startup_timeout_seconds: float = Field(default=1800, gt=0)

    @model_validator(mode="after")
    def validate_pools(self) -> Self:
        names = {pool.name for pool in self.pools}
        if len(names) != len(self.pools):
            raise ValueError("Each pool needs a unique name")
        from tokenflow.provisioner.cloud import CloudProvisioner
        from tokenflow.provisioner.slurm import SlurmProvisioner

        for pool in self.pools:
            adapter = SlurmProvisioner if pool.provider == "slurm" else CloudProvisioner
            adapter.validate_pool(pool)
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        config = cls.model_validate_json(path.read_bytes())
        config.state_path = (path.resolve().parent / config.state_path).resolve()
        return config

    def providers(self, worker_secret: str | None) -> dict[str, Provisioner]:
        from tokenflow.provisioner.cloud import CloudProvisioner
        from tokenflow.provisioner.slurm import SlurmProvisioner

        providers: dict[str, Provisioner] = {}
        kinds = {pool.provider for pool in self.pools}
        if "slurm" in kinds:
            if not worker_secret:
                raise ValueError("Managed Slurm pools require TOKENFLOW_WORKER_SECRET")
            providers["slurm"] = SlurmProvisioner(self.namespace, worker_secret)
        if "cloud" in kinds:
            providers["cloud"] = CloudProvisioner(self.namespace)
        return providers
