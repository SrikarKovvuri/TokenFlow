from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from tokenflow.config import Settings, get_settings


class ServiceHealth(BaseModel):
    status: str = "ok"
    service: str
    version: str


router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=ServiceHealth)
async def healthz(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ServiceHealth:
    return ServiceHealth(
        service=settings.app_name,
        version=settings.app_version,
    )
