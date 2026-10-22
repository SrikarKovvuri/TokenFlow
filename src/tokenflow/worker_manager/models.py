from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, StringConstraints, model_validator


WorkerName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]


class HealthStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class WorkerCreate(BaseModel):
    name: WorkerName
    base_url: AnyHttpUrl
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WorkerUpdate(BaseModel):
    name: WorkerName | None = None
    base_url: AnyHttpUrl | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

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
    registered_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="forbid", frozen=True)
