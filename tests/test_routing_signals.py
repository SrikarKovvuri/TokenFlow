import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from tokenflow.main import create_app
from tokenflow.scheduler.selection import NoAvailableWorkerError, WorkersAtCapacityError
from tokenflow.worker_manager.health_monitor import WorkerHealthMonitor
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate, WorkerTelemetry, WorkerUpdate
from tokenflow.worker_manager.registry import WorkerRegistry


async def add_worker(registry, name, **metadata):
    worker = await registry.create(WorkerCreate(
        name=name, base_url=f"http://{name}:8000",
        metadata={"model": "qwen", **metadata},
    ))
    return await registry.set_health_status(
        worker.id, HealthStatus.HEALTHY, expected_base_url=str(worker.base_url)
    )


@pytest.mark.parametrize("metrics", [
    'vllm:num_requests_running{engine="0"} 2\nvllm:num_requests_waiting 3\nvllm:kv_cache_usage_perc 0.7',
    'sglang:num_running_reqs{tp_rank="0"} 2\nsglang:num_running_reqs{tp_rank="1"} 2\n'
    'sglang:num_queue_reqs 3\nsglang:token_usage 0.7',
])
def test_monitor_collects_engine_and_physical_gpu_separately(metrics):
    async def scenario():
        registry = WorkerRegistry()
        worker = await add_worker(registry, "worker", backend="vllm", gpu_metrics_port=8001)

        def handler(request):
            if request.url.port == 8001:
                return httpx.Response(200, text=(
                    'DCGM_FI_DEV_GPU_UTIL{UUID="GPU-a"} 30\n'
                    'DCGM_FI_DEV_GPU_UTIL{UUID="GPU-b"} 72\n'
                    'DCGM_FI_DEV_GPU_UTIL{UUID="GPU-c"} NaN'
                ))
            return httpx.Response(200, text=metrics if request.url.path == "/metrics" else "")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await WorkerHealthMonitor(registry, client, interval_seconds=2, timeout_seconds=1).check_once()
        sample = (await registry.get(worker.id)).telemetry
        assert sample.running == 2
        assert sample.waiting == 3
        assert sample.kv_cache_usage == 0.7
        assert sample.gpu_utilization == 0.72
    asyncio.run(scenario())


def test_stale_telemetry_and_old_url_do_not_block_new_capacity():
    async def scenario():
        registry = WorkerRegistry()
        worker = await add_worker(registry, "worker", capacity=1)
        busy = WorkerTelemetry(observed_at=datetime.now(UTC), waiting=1)
        await registry.set_telemetry(worker.id, busy, expected_base_url=str(worker.base_url))
        with pytest.raises(WorkersAtCapacityError):
            await registry.assign("qwen")
        stale = busy.model_copy(update={"observed_at": busy.observed_at - timedelta(minutes=2)})
        await registry.set_telemetry(worker.id, stale, expected_base_url=str(worker.base_url))
        assert (await registry.assign("qwen")).id == worker.id
        await registry.release(worker.id)
        await registry.update(worker.id, WorkerUpdate(base_url="http://new:8000"))
        await registry.set_telemetry(worker.id, busy, expected_base_url=str(worker.base_url))
        assert (await registry.get(worker.id)).telemetry is None
    asyncio.run(scenario())


def test_heterogeneous_service_estimate_and_drain():
    async def scenario():
        registry = WorkerRegistry()
        slow = await add_worker(registry, "slow", service_time_seconds=4.0)
        fast = await add_worker(registry, "fast", service_time_seconds=1.0)
        first = await registry.assign("qwen")
        assert first.id == fast.id
        draining = await registry.update(fast.id, WorkerUpdate(enabled=False))
        assert draining.active_requests == 1
        assert (await registry.assign("qwen")).id == slow.id
        await registry.release(fast.id)
        assert (await registry.get(fast.id)).active_requests == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("first, second, chosen", [
    ({"waiting": 4}, {"running": 1}, "second"),
    ({"kv_cache_usage": 0.2}, {"kv_cache_usage": 0.9}, "first"),
    ({"gpu_utilization": 0.9}, {"gpu_utilization": 0.1}, "second"),
])
def test_fresh_pressure_signals_change_the_routing_decision(first, second, chosen):
    async def scenario():
        registry = WorkerRegistry()
        for name, gauges in (("first", first), ("second", second)):
            worker = await add_worker(registry, name)
            sample = WorkerTelemetry(observed_at=datetime.now(UTC), **gauges)
            await registry.set_telemetry(worker.id, sample, expected_base_url=str(worker.base_url))
        assert (await registry.assign("qwen")).name == chosen
    asyncio.run(scenario())


def test_at_capacity_returns_429_with_retry_hint():
    registry = WorkerRegistry()
    asyncio.run(add_worker(registry, "worker", capacity=1))
    lease = asyncio.run(registry.assign("qwen"))
    with TestClient(create_app(registry=registry, enable_health_monitor=False)) as client:
        response = client.post("/v1/chat/completions", json={"model": "qwen"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    asyncio.run(registry.release(lease.id))


def test_capacity_and_demand_include_rejected_requests_and_only_configured_models():
    async def scenario():
        registry = WorkerRegistry()
        registry.track_demand(["qwen"])
        with pytest.raises(NoAvailableWorkerError):
            await registry.assign("qwen")
        assert registry.demand_since("qwen", 15) == 1
        await add_worker(registry, "worker", capacity=1)
        await registry.assign("qwen")
        with pytest.raises(WorkersAtCapacityError):
            await registry.assign("qwen")
        assert registry.demand_since("qwen", 15) == 2
        with pytest.raises(NoAvailableWorkerError):
            await registry.assign("unknown-model")
        assert registry.demand_since("unknown-model", 15) == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("failure,code,expected_hosts", [
    (httpx.ConnectError, 200, ["primary", "backup"]),
    (httpx.ReadTimeout, 504, ["primary"]),
])
def test_failover_only_on_connection_setup(failure, code, expected_hosts):
    registry = WorkerRegistry()
    asyncio.run(add_worker(registry, "primary"))
    asyncio.run(add_worker(registry, "backup"))
    hosts = []

    def handle(request):
        hosts.append(request.url.host)
        if request.url.host == "primary":
            raise failure("unavailable", request=request)
        return httpx.Response(200, json={"choices": []})

    with TestClient(create_app(
        registry=registry, worker_transport=httpx.MockTransport(handle), enable_health_monitor=False,
    )) as client:
        assert client.post("/v1/chat/completions", json={"model": "qwen"}).status_code == code
    assert hosts == expected_hosts
    assert all(w.active_requests == 0 for w in asyncio.run(registry.list()))
