import sys
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID

from pydantic import AfterValidator, AnyHttpUrl, BaseModel, ConfigDict, Field, StringConstraints, model_validator


WorkerName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]
DEFAULT_CAPACITY = 32


def _validate_scheduling_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    capacity = metadata.get("capacity", DEFAULT_CAPACITY)
    if type(capacity) is not int or capacity < 1:
        raise ValueError("metadata.capacity must be a positive integer")
    seconds = metadata.get("service_time_seconds", 1.0)
    if type(seconds) not in (int, float) or not 0 < seconds <= sys.float_info.max:
        raise ValueError("metadata.service_time_seconds must be a positive finite number")
    return metadata


WorkerMetadata = Annotated[dict[str, Any], AfterValidator(_validate_scheduling_metadata)]


class HealthStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class WorkerTelemetry(BaseModel):
    """Latest sampled gauges. Utilization and cache occupancy are fractions."""

    observed_at: datetime
    running: int | None = Field(default=None, ge=0)
    waiting: int | None = Field(default=None, ge=0)
    gateway_overlap: int = Field(default=0, ge=0)
    kv_cache_usage: float | None = Field(default=None, ge=0, le=1)
    gpu_utilization: float | None = Field(default=None, ge=0, le=1)

    model_config = ConfigDict(frozen=True)

    @property
    def requests(self) -> int:
        return (self.running or 0) + (self.waiting or 0)


class WorkerCreate(BaseModel):
    name: WorkerName
    base_url: AnyHttpUrl
    enabled: bool = True
    metadata: WorkerMetadata = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WorkerUpdate(BaseModel):
    name: WorkerName | None = None
    base_url: AnyHttpUrl | None = None
    enabled: bool | None = None
    metadata: WorkerMetadata | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def require_non_null_update(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")

        null_fields = {
            field
            for field in self.model_fields_set
            if getattr(self, field) is None
        }
        if null_fields:
            fields = ", ".join(sorted(null_fields))
            raise ValueError(f"Fields cannot be null: {fields}")

        return self


class Worker(WorkerCreate):
    id: UUID
    health_status: HealthStatus = HealthStatus.UNKNOWN
    active_requests: int = 0
    telemetry: WorkerTelemetry | None = None
    registered_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def capacity(self) -> int:
        return self.metadata.get("capacity", DEFAULT_CAPACITY)

    def gateway_capacity(self, sample: WorkerTelemetry | None) -> int:
        """Total slots for gateway leases after allowing for sampled external load."""
        external = max(0, sample.requests - sample.gateway_overlap) if sample else 0
        return max(0, self.capacity - external)

    def fresh_telemetry(self, now: datetime, max_age_seconds: float) -> WorkerTelemetry | None:
        if self.telemetry and (now - self.telemetry.observed_at).total_seconds() <= max_age_seconds:
            return self.telemetry
        return None
