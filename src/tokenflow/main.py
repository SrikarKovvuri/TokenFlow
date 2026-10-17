from uuid import uuid4

from fastapi import FastAPI, Request, Response

from tokenflow.api.health import router as health_router
from tokenflow.api.workers import router as workers_router
from tokenflow.config import Settings, get_settings
from tokenflow.registry import WorkerRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Distributed LLM inference gateway and control plane.",
    )
    application.state.worker_registry = WorkerRegistry()

    @application.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    application.include_router(health_router)
    application.include_router(workers_router)
    return application


app = create_app()
