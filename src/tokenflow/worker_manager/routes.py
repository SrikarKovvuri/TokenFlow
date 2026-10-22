from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status

from tokenflow.worker_manager.models import Worker, WorkerCreate, WorkerUpdate
from tokenflow.worker_manager.registry import (
    DuplicateWorkerError,
    WorkerNotFoundError,
)


router = APIRouter(prefix="/v1/workers", tags=["workers"])


@router.post("", response_model=Worker, status_code=status.HTTP_201_CREATED)
async def create_worker(
    worker_data: WorkerCreate,
    request: Request,
    response: Response,
) -> Worker:
    try:
        worker = await request.app.state.worker_registry.create(worker_data)
    except DuplicateWorkerError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    response.headers["Location"] = f"/v1/workers/{worker.id}"
    return worker


@router.get("", response_model=list[Worker])
async def list_workers(request: Request) -> list[Worker]:
    return await request.app.state.worker_registry.list()


@router.get("/{worker_id}", response_model=Worker)
async def get_worker(worker_id: UUID, request: Request) -> Worker:
    try:
        return await request.app.state.worker_registry.get(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.patch("/{worker_id}", response_model=Worker)
async def update_worker(
    worker_id: UUID,
    worker_data: WorkerUpdate,
    request: Request,
) -> Worker:
    try:
        return await request.app.state.worker_registry.update(
            worker_id,
            worker_data,
        )
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
async def delete_worker(worker_id: UUID, request: Request) -> Response:
    try:
        await request.app.state.worker_registry.delete(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
