import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from tokenflow.scheduler.proxy import forward_chat_completion
from tokenflow.scheduler.selection import NoAvailableWorkerError, select_first_healthy


router = APIRouter(prefix="/v1", tags=["inference"])
logger = logging.getLogger("uvicorn.error")


@router.post("/chat/completions")
async def create_chat_completion(
    payload: dict[str, Any],
    request: Request,
) -> Response:
    worker_manager = request.app.state.worker_registry
    try:
        workers = await worker_manager.list()
        worker = select_first_healthy(workers)
    except NoAvailableWorkerError as exc:
        logger.info("No healthy worker among %d registered workers", len(workers))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    logger.info(
        "Fetched %d workers; routed to %s (%s) at %s",
        len(workers),
        worker.name,
        worker.id,
        worker.base_url,
    )
    return await forward_chat_completion(
        worker=worker,
        payload=payload,
        request_id=request.state.request_id,
        client=request.app.state.worker_client,
    )
