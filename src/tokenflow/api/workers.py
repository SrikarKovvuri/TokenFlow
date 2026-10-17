from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tokenflow.api.dependencies import get_registry
from tokenflow.models import Worker, WorkerCreate, WorkerUpdate
from tokenflow.registry import (
    DuplicateWorkerError,
    WorkerNotFoundError,
    WorkerRegistry,
)


router = APIRouter(prefix="/v1/workers", tags=["workers"])
Registry = Annotated[WorkerRegistry, Depends(get_registry)]


@router.post("", response_model=Worker, status_code=status.HTTP_201_CREATED)
async def create_worker(
    worker_data: WorkerCreate,
    response: Response,
    registry: Registry,
) -> Worker:
    try:
        worker = await registry.create(worker_data)
    except DuplicateWorkerError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    response.headers["Location"] = f"/v1/workers/{worker.id}"
    return worker


@router.get("", response_model=list[Worker])
async def list_workers(registry: Registry) -> list[Worker]:
    return await registry.list()


@router.get("/{worker_id}", response_model=Worker)
async def get_worker(worker_id: UUID, registry: Registry) -> Worker:
    try:
        return await registry.get(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.patch("/{worker_id}", response_model=Worker)
async def update_worker(
    worker_id: UUID,
    worker_data: WorkerUpdate,
    registry: Registry,
) -> Worker:
    try:
        return await registry.update(worker_id, worker_data)
    except WorkerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DuplicateWorkerError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{worker_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_worker(worker_id: UUID, registry: Registry) -> Response:
    try:
        await registry.delete(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
