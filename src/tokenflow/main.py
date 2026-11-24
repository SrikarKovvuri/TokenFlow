import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Request, Response

from tokenflow.config import Settings, get_settings
from tokenflow.platform import api as platform_api
from tokenflow.platform.auth import require_key
from tokenflow.platform.config import PlatformConfig
from tokenflow.platform.lease import controller_lease
from tokenflow.provisioner.base import Provisioner
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
    providers: dict[str, Provisioner] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    registry = registry or WorkerRegistry()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            settings.worker_request_timeout_seconds,
            connect=settings.worker_connect_timeout_seconds,
        )
        async with AsyncExitStack() as stack:
            tasks: list[asyncio.Task] = []
            hooks = {}
            if settings.platform_config:
                from tokenflow.platform.controller import Controller

                config = PlatformConfig.load(settings.platform_config)
                secret = settings.worker_secret.get_secret_value() if settings.worker_secret else None
                adapters = config.providers(secret) if providers is None else providers
                stack.enter_context(controller_lease(config.state_path, config.namespace))
                controller = Controller(
                    config.pools, adapters, registry, config.state_path,
                    interval_seconds=config.interval_seconds,
                    idle_seconds=config.idle_seconds,
                    startup_timeout_seconds=config.startup_timeout_seconds,
                )
                stack.callback(controller.close)
                application.state.controller = controller
                application.state.platform_config = config
                if secret:
                    from tokenflow.provisioner.slurm import worker_token

                    async def authenticate_worker(request: httpx.Request) -> None:
                        key = controller.slurm_key_for_url(request.url)
                        if key:
                            request.headers["Authorization"] = f"Bearer {worker_token(secret, key)}"

                    hooks["request"] = [authenticate_worker]

            client_options = dict(transport=worker_transport, trust_env=False, event_hooks=hooks)
            worker_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout, **client_options)
            )
            # A fresh connection per probe avoids stale keep-alive sockets across
            # SSH tunnels and keeps health polling independent of inference traffic.
            health_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    timeout=settings.worker_health_timeout_seconds,
                    limits=httpx.Limits(max_keepalive_connections=0), **client_options,
                )
            )
            application.state.worker_client = worker_client
            if enable_health_monitor:
                monitor = WorkerHealthMonitor(
                    registry,
                    health_client,
                    interval_seconds=settings.worker_health_interval_seconds,
                    timeout_seconds=settings.worker_health_timeout_seconds,
                )
                tasks.append(asyncio.create_task(monitor.run(), name="health-monitor"))
            if application.state.controller:
                tasks.append(asyncio.create_task(application.state.controller.run(), name="controller"))

            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError):
                        await task
                application.state.controller = None
                application.state.platform_config = None

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Distributed LLM inference gateway and control plane.",
        lifespan=lifespan,
    )
    application.state.worker_registry = registry
    application.state.platform_config = None
    application.state.controller = None

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

    admin = [Depends(require_key([settings.admin_key] if settings.admin_key else []))]
    student = [Depends(require_key(settings.api_keys))]
    application.include_router(worker_router, dependencies=admin)
    application.include_router(platform_api.admin_router, dependencies=admin)
    application.include_router(scheduler_router, dependencies=student)
    application.include_router(platform_api.router, dependencies=student)
    return application


app = create_app()
