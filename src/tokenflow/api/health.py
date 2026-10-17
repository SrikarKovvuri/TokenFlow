from typing import Annotated

from fastapi import APIRouter, Depends

from tokenflow.config import Settings, get_settings
from tokenflow.models import ServiceHealth


router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=ServiceHealth)
async def healthz(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ServiceHealth:
    return ServiceHealth(
        service=settings.app_name,
        version=settings.app_version,
    )
