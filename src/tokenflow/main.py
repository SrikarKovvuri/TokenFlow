import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response

from tokenflow.api.health import router as health_router
from tokenflow.api.inference import router as inference_router
from tokenflow.api.workers import router as workers_router
from tokenflow.config import Settings, get_settings
from tokenflow.health_monitor import WorkerHealthMonitor
from tokenflow.registry import WorkerRegistry


def create_app(
    settings: Settings | None = None,
    *,
    registry: WorkerRegistry | None = None,
    worker_transport: httpx.AsyncBaseTransport | None = None,
    enable_health_monitor: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    registry = registry or WorkerRegistry()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            settings.worker_request_timeout_seconds,
            connect=settings.worker_connect_timeout_seconds,
        )
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=worker_transport,
        ) as worker_client:
            application.state.worker_client = worker_client
            monitor = WorkerHealthMonitor(
                registry,
                worker_client,
                interval_seconds=settings.worker_health_interval_seconds,
                timeout_seconds=settings.worker_health_timeout_seconds,
            )
            monitor_task: asyncio.Task[None] | None = None
            if enable_health_monitor:
                monitor_task = asyncio.create_task(monitor.run())

            try:
                yield
            finally:
                if monitor_task is not None:
                    monitor_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await monitor_task

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Distributed LLM inference gateway and control plane.",
        lifespan=lifespan,
    )
    application.state.worker_registry = registry

    @application.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    application.include_router(health_router)
    application.include_router(workers_router)
    application.include_router(inference_router)
    return application


app = create_app()
