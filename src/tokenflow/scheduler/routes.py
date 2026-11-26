import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from tokenflow.scheduler.body_limit import InferenceRoute
from tokenflow.scheduler.proxy import WorkerConnectionError, forward_chat_completion
from tokenflow.scheduler.selection import NoAvailableWorkerError, WorkersAtCapacityError
from tokenflow.worker_manager.models import HealthStatus


router = APIRouter(prefix="/v1", tags=["inference"], route_class=InferenceRoute)
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

    choices = payload.get("n")
    if choices is None:
        choices = 1
    if type(choices) is not int or choices < 1:
        return JSONResponse(status_code=400, content={"error": {
            "message": "n must be a positive integer.", "type": "invalid_request_error",
            "param": "n", "code": "invalid_value",
        }})

    worker_manager = request.app.state.worker_registry
    excluded = set()
    connection_failure = None
    while True:
        try:
            worker = await worker_manager.assign(model, choices=choices, exclude=excluded)
        except (NoAvailableWorkerError, WorkersAtCapacityError) as exc:
            if connection_failure is not None:
                raise connection_failure
            code = 429 if isinstance(exc, WorkersAtCapacityError) else 503
            headers = {"Retry-After": "1"}
            controller = request.app.state.controller
            if code == 503 and controller is not None and controller.accepts_cold_demand(model):
                headers["X-TokenFlow-Admission"] = "waiting-for-worker"
            raise HTTPException(code, str(exc), headers=headers) from exc

        logger.info("Routed model %s to %s with %d active requests",
                    model, worker.id, worker.active_requests)
        endpoint = worker_manager.endpoint_identity(worker.id)
        try:
            return await forward_chat_completion(
                worker=worker,
                payload=payload,
                request_id=request.state.request_id,
                client=request.app.state.worker_client,
                release=worker_manager.release_callback(worker.id, choices=choices),
                receive=request.receive,
            )
        except WorkerConnectionError as exc:
            await worker_manager.set_health_status(
                worker.id, HealthStatus.UNHEALTHY, expected_base_url=str(worker.base_url), expected_endpoint=endpoint,
            )
            # One failover, only before a connection was established. A read
            # timeout or interrupted SSE stream may already have generated tokens.
            if excluded:
                raise
            excluded.add(worker.id)
            connection_failure = exc
