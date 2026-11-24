from typing import Self

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tokenflow.worker_manager.models import HealthStatus


router = APIRouter()
admin_router = APIRouter()


class ReplicaLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("minimum exceeds maximum")
        return self


@router.get("/v1/models", tags=["inference"])
async def models(request: Request) -> dict:
    registry = request.app.state.worker_registry
    names = {
        worker.metadata["model"]
        for worker in await registry.list()
        if worker.enabled and worker.health_status == HealthStatus.HEALTHY
        and isinstance(worker.metadata.get("model"), str)
    }
    config = request.app.state.platform_config
    if config:
        names.update(pool.model for pool in config.pools)
    return {"object": "list", "data": [
        {"id": name, "object": "model", "created": 0, "owned_by": "tokenflow"}
        for name in sorted(names)
    ]}


def _controller(request: Request):
    controller = request.app.state.controller
    if controller is None:
        raise HTTPException(404, "Managed pools are disabled")
    return controller


@admin_router.get("/v1/pools", tags=["administration"])
async def pools(request: Request) -> list[dict]:
    return _controller(request).status()


@admin_router.put("/v1/pools/{name}/limits", tags=["administration"])
async def set_pool_limits(name: str, limits: ReplicaLimits, request: Request) -> dict:
    controller = _controller(request)
    try:
        controller.set_limits(name, limits.minimum, limits.maximum)
    except KeyError as exc:
        raise HTTPException(404, "Unknown pool") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"pool": name, **limits.model_dump()}
