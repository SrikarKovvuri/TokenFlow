from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from tokenflow.scheduler.proxy import forward_chat_completion
from tokenflow.scheduler.selection import NoAvailableWorkerError, select_first_healthy


router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/chat/completions")
async def create_chat_completion(
    payload: dict[str, Any],
    request: Request,
) -> Response:
    try:
        worker = await select_first_healthy(request.app.state.worker_registry)
    except NoAvailableWorkerError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return await forward_chat_completion(
        worker=worker,
        payload=payload,
        request_id=request.state.request_id,
        client=request.app.state.worker_client,
    )
