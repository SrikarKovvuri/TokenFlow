import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response

from tokenflow.config import Settings, get_settings
from tokenflow.scheduler.routes import router as scheduler_router
from tokenflow.worker_manager.health_monitor import WorkerHealthMonitor
from tokenflow.worker_manager.registry import WorkerRegistry
from tokenflow.worker_manager.routes import router as worker_router


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
        async with (
            httpx.AsyncClient(
                timeout=timeout,
                transport=worker_transport,
            ) as worker_client,
            # A fresh connection per probe avoids stale keep-alive sockets across
            # SSH tunnels and keeps health polling independent of inference traffic.
            httpx.AsyncClient(
                timeout=settings.worker_health_timeout_seconds,
                transport=worker_transport,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as health_client,
        ):
            application.state.worker_client = worker_client
            monitor_task: asyncio.Task[None] | None = None
            if enable_health_monitor:
                monitor = WorkerHealthMonitor(
                    registry,
                    health_client,
                    interval_seconds=settings.worker_health_interval_seconds,
                    timeout_seconds=settings.worker_health_timeout_seconds,
                )
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

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
        }

    application.include_router(worker_router)
    application.include_router(scheduler_router)
    return application


app = create_app()
