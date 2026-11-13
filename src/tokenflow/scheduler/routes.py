import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from tokenflow.scheduler.proxy import forward_chat_completion
from tokenflow.scheduler.selection import NoAvailableWorkerError


router = APIRouter(prefix="/v1", tags=["inference"])
logger = logging.getLogger("uvicorn.error")


@router.post("/chat/completions")
async def create_chat_completion(
    payload: dict[str, Any],
    request: Request,
) -> Response:
    model = payload.get("model")
    if not isinstance(model, str) or model == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A model is required",
        )

    worker_manager = request.app.state.worker_registry
    try:
        worker = await worker_manager.assign(model)
    except NoAvailableWorkerError as exc:
        logger.info("No eligible worker for model %s", model)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    logger.info(
        "Routed model %s to %s (%s) at %s with %d active requests",
        model,
        worker.name,
        worker.id,
        worker.base_url,
        worker.active_requests,
    )
    return await forward_chat_completion(
        worker=worker,
        payload=payload,
        request_id=request.state.request_id,
        client=request.app.state.worker_client,
        release=lambda: worker_manager.release(worker.id),
    )
