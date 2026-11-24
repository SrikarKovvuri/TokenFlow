"""Service boundaries shared by the controller and the two GPU providers."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,31}$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Backend = Literal["vllm", "sglang"]
Provider = Literal["slurm", "cloud"]


class PoolSpec(BaseModel):
    """An operator-approved hardware and checkpoint combination."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: Identifier
    model: Text
    checkpoint: Text
    backend: Backend = "vllm"
    provider: Provider
    gpu_type: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]
    gpu_count: int = Field(default=1, ge=1)
    image: Text
    port: int = Field(default=8000, ge=1024, le=65534)
    capacity: int = Field(default=16, ge=1)
    service_time_seconds: float = Field(default=1, gt=0)
    min_replicas: int = Field(default=0, ge=0)
    max_replicas: int = Field(ge=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas exceeds max_replicas")
        return self


@dataclass(frozen=True)
class Resource:
    """A provider allocation. A running allocation still needs an HTTP health check."""

    key: str
    external_id: str
    phase: Literal["pending", "running", "terminal"]
    base_url: str | None = None


class Provisioner(Protocol):
    async def discover(self, pool: PoolSpec) -> list[Resource]: ...

    async def launch(self, pool: PoolSpec, key: str) -> Resource: ...

    async def terminate(self, pool: PoolSpec, resource: Resource) -> None: ...
