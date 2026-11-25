import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

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


def test_engine_load_leaves_one_slot_for_a_burst_until_the_next_sample():
    async def scenario():
        registry = WorkerRegistry()
        fast = await add_worker(registry, "h100", capacity=4, service_time_seconds=1)
        slow = await add_worker(registry, "a6000", capacity=4, service_time_seconds=3)
        url = str(fast.base_url)
        sample = WorkerTelemetry(observed_at=datetime.now(UTC), running=2, waiting=1)
        await registry.set_telemetry(fast.id, sample, expected_base_url=url)
        assert registry.capacity_for_gateway(await registry.get(fast.id)) == 1
        first, second = await asyncio.gather(registry.assign("qwen"), registry.assign("qwen"))
        assert (first.id, second.id) == (fast.id, slow.id)
        assert registry.capacity_for_gateway(first) == 1
        # The refresh observes this gateway's lease among four engine requests.
        await registry.set_telemetry(fast.id, sample.model_copy(update={"running": 3}), expected_base_url=url)
        await registry.release(first.id)
        assert registry.capacity_for_gateway(await registry.get(fast.id)) == 1
        assert (await registry.assign("qwen")).id == fast.id
        assert (await registry.assign("qwen")).id == slow.id
        stale = sample.model_copy(update={"observed_at": datetime.now(UTC) - timedelta(minutes=1)})
        await registry.set_telemetry(fast.id, stale, expected_base_url=url)
        assert registry.capacity_for_gateway(await registry.get(fast.id)) == 4
        assert (await registry.assign("qwen")).id == fast.id
    asyncio.run(scenario())


def test_assign_and_release_during_network_probe_cannot_hide_external_work():
    async def scenario():
        registry = WorkerRegistry()
        worker = await add_worker(registry, "l40s", backend="sglang", capacity=4)
        await registry.assign("qwen")
        reading, finish = asyncio.Event(), asyncio.Event()

        async def server(request):
            if request.url.path == "/metrics":
                reading.set()
                await finish.wait()
                return httpx.Response(200, text="sglang:num_running_reqs 3")
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client, asyncio.timeout(2):
            monitor = WorkerHealthMonitor(registry, client, interval_seconds=1, timeout_seconds=1)
            polling = asyncio.create_task(monitor.check_once())
            await reading.wait()
            await registry.release(worker.id)
            await registry.assign("qwen")
            finish.set()
            await polling
        current = await registry.get(worker.id)
        assert current.telemetry.gateway_overlap == 0 and current.active_requests == 1
        assert registry.capacity_for_gateway(current) == 1
        with pytest.raises(WorkersAtCapacityError):
            await registry.assign("qwen")
    asyncio.run(scenario())


def test_superseded_probes_and_old_endpoint_reservations_do_not_mask_new_load():
    async def scenario():
        registry = WorkerRegistry()
        old = await add_worker(registry, "first", capacity=4)
        await registry.assign("qwen")
        stale = registry.begin_telemetry_probe(old.id)
        old_url = str(old.base_url)
        await registry.update(old.id, WorkerUpdate(base_url="http://other"))
        await registry.update(old.id, WorkerUpdate(base_url=old_url))
        await registry.set_health_status(old.id, HealthStatus.HEALTHY, expected_base_url=old_url)
        busy = WorkerTelemetry(observed_at=datetime.now(UTC), running=3)
        await registry.set_telemetry(old.id, busy, expected_base_url=old_url, probe=stale)
        assert (await registry.get(old.id)).telemetry is None
        earlier = registry.begin_telemetry_probe(old.id)
        newer = registry.begin_telemetry_probe(old.id)
        await registry.set_telemetry(old.id, busy, expected_base_url=old_url, probe=newer)
        await registry.set_telemetry(old.id, None, expected_base_url=old_url, probe=earlier)
        current = await registry.get(old.id)
        assert current.telemetry.gateway_overlap == 0
        assert registry.capacity_for_gateway(current) == 1
        await registry.release(old.id)
        assert (await registry.assign("qwen")).id == old.id
        with pytest.raises(WorkersAtCapacityError):
            await registry.assign("qwen")
    asyncio.run(scenario())


def test_old_stream_cannot_release_a_replacement_registered_in_the_same_clock_tick(monkeypatch):
    now = datetime.now(UTC)
    monkeypatch.setattr("tokenflow.worker_manager.registry.datetime", SimpleNamespace(now=lambda tz: now))

    async def scenario():
        registry = WorkerRegistry()
        old = await add_worker(registry, "first", capacity=1)
        await registry.assign("qwen")
        release_old_stream = registry.release_callback(old.id)
        await registry.delete(old.id)
        replacement = await registry.create(
            WorkerCreate(name="restored", base_url=old.base_url, metadata={"model": "qwen", "capacity": 1}),
            worker_id=old.id,
        )
        assert replacement.registered_at == old.registered_at
        await registry.set_health_status(replacement.id, HealthStatus.HEALTHY, expected_base_url=str(old.base_url))
        await registry.assign("qwen")
        release_new_stream = registry.release_callback(replacement.id)
        await release_old_stream()
        assert (await registry.get(replacement.id)).active_requests == 1
        await release_new_stream()
        assert (await registry.get(replacement.id)).active_requests == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["health-poll", "inference-connection"])
@pytest.mark.parametrize("replacement", ["same-id", "url-changed-back"])
def test_health_write_ignores_a_replacement_at_the_same_address(source, replacement):
    async def scenario():
        registry = WorkerRegistry()
        old = await add_worker(registry, "first", capacity=1)
        url = str(old.base_url)

        async def upstream(request):
            if replacement == "same-id":
                await registry.delete(old.id)
                await registry.create(WorkerCreate(name="new", base_url=url,
                                      metadata={"model": "qwen", "capacity": 1}), worker_id=old.id)
            else:
                await registry.update(old.id, WorkerUpdate(base_url="http://other"))
                await registry.update(old.id, WorkerUpdate(base_url=url))
            if source == "health-poll":
                assert request.url.path == "/health"
                return httpx.Response(200)
            assert request.url.path == "/v1/chat/completions"
            await registry.set_health_status(old.id, HealthStatus.HEALTHY, expected_base_url=url)
            raise httpx.ConnectError("old worker gone", request=request)

        transport = httpx.MockTransport(upstream)
        if source == "health-poll":
            async with httpx.AsyncClient(transport=transport) as client:
                await WorkerHealthMonitor(registry, client, interval_seconds=1, timeout_seconds=1).check_once()
            assert (await registry.get(old.id)).health_status is HealthStatus.UNKNOWN
        else:
            app = create_app(registry=registry, worker_transport=transport, enable_health_monitor=False)
            async with (app.router.lifespan_context(app),
                        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client):
                result = await client.post("/v1/chat/completions", json={"model": "qwen"})
            assert result.status_code == 502
            current = await registry.get(old.id)
            assert current.health_status is HealthStatus.HEALTHY and current.active_requests == 0

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
