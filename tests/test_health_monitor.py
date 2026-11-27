import asyncio

import httpx
import pytest

from tokenflow.scheduler.selection import NoAvailableWorkerError
from tokenflow.worker_manager.health_monitor import WorkerHealthMonitor
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate, WorkerUpdate
from tokenflow.worker_manager.registry import WorkerRegistry


@pytest.mark.parametrize("status_code", [200, 503])
def test_health_result_is_ignored_when_url_changes(status_code: int) -> None:
    async def scenario() -> None:
        registry = WorkerRegistry()
        worker = await registry.create(
            WorkerCreate(name="worker", base_url="http://old-worker:8101")
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "old-worker"
            await registry.update(
                worker.id, WorkerUpdate(base_url="http://new-worker:8101")
            )
            return httpx.Response(status_code)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monitor = WorkerHealthMonitor(
                registry, client, interval_seconds=5, timeout_seconds=1
            )
            await monitor.check_once()

        updated = await registry.get(worker.id)
        assert str(updated.base_url) == "http://new-worker:8101/"
        assert updated.health_status is HealthStatus.UNKNOWN

    asyncio.run(scenario())


def test_reapplying_worker_address_keeps_it_routable_during_metrics_scrape() -> None:
    async def scenario() -> None:
        registry = WorkerRegistry()
        worker = await registry.create(WorkerCreate(
            name="worker", base_url="http://worker",
            metadata={"model": "qwen", "backend": "vllm"},
        ))

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200)
            assert request.url.path == "/metrics"
            # A declarative client reapplies the same normalized URL while the
            # monitor is waiting on this worker's scrape.
            await registry.update(worker.id, WorkerUpdate(base_url="http://WORKER:80/"))
            assigned = await registry.assign("qwen")
            await registry.release(assigned.id)
            return httpx.Response(200, text="vllm:num_requests_running 2\n")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await WorkerHealthMonitor(registry, client, interval_seconds=5, timeout_seconds=1).check_once()

        current = await registry.get(worker.id)
        assert current.health_status is HealthStatus.HEALTHY
        assert current.telemetry.running == 2

    asyncio.run(scenario())


def test_health_check_marks_responsive_worker_healthy() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health"
            return httpx.Response(200)

        registry = WorkerRegistry()
        worker = await registry.create(
            WorkerCreate(
                name="worker-1",
                base_url="http://worker:8101",
                metadata={"model": "test-model"},
            )
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
        chosen = await registry.assign("test-model")
        assert chosen.id == worker.id
        await registry.release(chosen.id)

        await registry.update(worker.id, WorkerUpdate(enabled=False))
        with pytest.raises(NoAvailableWorkerError):
            await registry.assign("test-model")

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
            WorkerCreate(
                name="worker-1",
                base_url="http://worker:8101",
                metadata={"model": "test-model"},
            )
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
            await registry.assign("test-model")

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


def test_health_transport_errors_are_logged(caplog):
    async def scenario():
        registry = WorkerRegistry()
        await registry.create(WorkerCreate(name="worker", base_url="http://worker"))
        async def handler(request):
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monitor = WorkerHealthMonitor(registry, client, interval_seconds=5, timeout_seconds=1)
            await monitor.check_once()
        assert (await registry.list())[0].health_status is HealthStatus.UNHEALTHY
    asyncio.run(scenario())
    assert "RemoteProtocolError" in caplog.text
    assert "Server disconnected" in caplog.text


def test_managed_worker_requires_its_advertised_model_and_recovers():
    async def scenario():
        registry = WorkerRegistry()
        worker = await registry.create(WorkerCreate(
            name="managed", base_url="http://worker", metadata={"model": "qwen", "backend": "sglang"},
        ))
        catalog = {"data": [{"id": "another-model"}]}

        def server(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json=catalog)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="sglang:num_queue_reqs 1e100\nsglang:token_usage 0.4")
            assert request.url.path == "/health"
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            monitor = WorkerHealthMonitor(registry, client, interval_seconds=5, timeout_seconds=1,
                                          managed_worker=lambda wid: wid == worker.id)
            await monitor.check_once()
            with pytest.raises(NoAvailableWorkerError):
                await registry.assign("qwen")
            catalog["data"] = [{"id": "qwen"}]
            await monitor.check_once()
            lease = await registry.assign("qwen")
            assert lease.id == worker.id
            assert lease.telemetry.waiting is None
            assert lease.telemetry.kv_cache_usage == 0.4
            await registry.release(lease.id)
            catalog["data"] = []
            await monitor.check_once()
            assert (await registry.get(worker.id)).health_status is HealthStatus.UNHEALTHY

    asyncio.run(scenario())


def test_stalled_health_endpoint_does_not_hold_up_later_polls():
    async def scenario():
        registry = WorkerRegistry()
        slow = await registry.create(WorkerCreate(name="slow", base_url="http://slow"))
        healthy = await registry.create(WorkerCreate(name="healthy", base_url="http://healthy"))
        healthy_statuses = iter((200, 503))

        async def server(request):
            if request.url.host == "slow":
                await asyncio.Event().wait()
            return httpx.Response(next(healthy_statuses))

        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            monitor = WorkerHealthMonitor(registry, client, interval_seconds=.01, timeout_seconds=.02)
            async with asyncio.timeout(3):
                await monitor.check_once()
                assert (await registry.get(slow.id)).health_status is HealthStatus.UNHEALTHY
                assert (await registry.get(healthy.id)).health_status is HealthStatus.HEALTHY
                await monitor.check_once()
                assert (await registry.get(healthy.id)).health_status is HealthStatus.UNHEALTHY

    asyncio.run(scenario())
