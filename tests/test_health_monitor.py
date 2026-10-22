import asyncio

import httpx
import pytest

from tokenflow.scheduler.selection import NoAvailableWorkerError, select_first_healthy
from tokenflow.worker_manager.health_monitor import WorkerHealthMonitor
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate, WorkerUpdate
from tokenflow.worker_manager.registry import WorkerRegistry


def test_health_check_marks_responsive_worker_healthy() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health"
            return httpx.Response(200)

        registry = WorkerRegistry()
        worker = await registry.create(
            WorkerCreate(name="worker-1", base_url="http://worker:8101")
        )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            monitor = WorkerHealthMonitor(
                registry,
                client,
                interval_seconds=5,
                timeout_seconds=1,
            )
            await monitor.check_once()

        updated = await registry.get(worker.id)
        assert updated.health_status is HealthStatus.HEALTHY
        assert (await select_first_healthy(registry)).id == worker.id

        await registry.update(worker.id, WorkerUpdate(enabled=False))
        with pytest.raises(NoAvailableWorkerError):
            await select_first_healthy(registry)

        reenabled = await registry.update(
            worker.id,
            WorkerUpdate(enabled=True),
        )
        assert reenabled.health_status is HealthStatus.UNKNOWN

    asyncio.run(scenario())


def test_health_check_marks_unreachable_worker_unhealthy() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("worker is offline", request=request)

        registry = WorkerRegistry()
        worker = await registry.create(
            WorkerCreate(name="worker-1", base_url="http://worker:8101")
        )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            monitor = WorkerHealthMonitor(
                registry,
                client,
                interval_seconds=5,
                timeout_seconds=1,
            )
            await monitor.check_once()

        updated = await registry.get(worker.id)
        assert updated.health_status is HealthStatus.UNHEALTHY
        with pytest.raises(NoAvailableWorkerError):
            await select_first_healthy(registry)

    asyncio.run(scenario())


def test_disabled_worker_is_not_health_checked() -> None:
    async def scenario() -> None:
        async def unexpected_handler(
            request: httpx.Request,
        ) -> httpx.Response:
            raise AssertionError(f"Unexpected request to {request.url}")

        registry = WorkerRegistry()
        worker = await registry.create(
            WorkerCreate(
                name="worker-1",
                base_url="http://worker:8101",
                enabled=False,
            )
        )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_handler)
        ) as client:
            monitor = WorkerHealthMonitor(
                registry,
                client,
                interval_seconds=5,
                timeout_seconds=1,
            )
            await monitor.check_once()

        updated = await registry.get(worker.id)
        assert updated.health_status is HealthStatus.UNKNOWN

    asyncio.run(scenario())
