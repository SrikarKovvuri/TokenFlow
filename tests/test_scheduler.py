import asyncio

import pytest

from tokenflow.scheduler.selection import NoAvailableWorkerError
from tokenflow.worker_manager.models import HealthStatus, Worker, WorkerCreate, WorkerUpdate
from tokenflow.worker_manager.registry import WorkerRegistry


def test_requests_stay_inside_one_model_pool() -> None:
    async def scenario() -> None:
        registry = WorkerRegistry()
        small = await _healthy(
            registry, "small", "http://small:8101", "qwen-0.5b"
        )
        large = await _healthy(
            registry, "large", "http://large:8102", "qwen-14b"
        )

        assert (await registry.assign("qwen-14b")).id == large.id
        assert (await registry.assign("qwen-0.5b")).id == small.id
        with pytest.raises(NoAvailableWorkerError):
            await registry.assign("qwen-32b")

    asyncio.run(scenario())


def test_least_active_worker_is_reserved_until_release() -> None:
    async def scenario() -> None:
        registry = WorkerRegistry()
        first = await _healthy(
            registry, "a", "http://a:8101", "qwen-0.5b"
        )
        second = await _healthy(
            registry, "b", "http://b:8102", "qwen-0.5b"
        )

        initial = await registry.assign("qwen-0.5b")
        diverted = await registry.assign("qwen-0.5b")
        assert initial.id == first.id
        assert diverted.id == second.id
        assert (await registry.get(first.id)).active_requests == 1
        assert (await registry.get(second.id)).active_requests == 1

        await registry.release(initial.id)
        assert (await registry.assign("qwen-0.5b")).id == first.id
        await registry.release(first.id)
        await registry.release(second.id)
        assert (await registry.get(first.id)).active_requests == 0

    asyncio.run(scenario())


def test_disabled_or_unhealthy_workers_are_not_ready() -> None:
    async def scenario() -> None:
        registry = WorkerRegistry()
        worker = await _healthy(
            registry, "a", "http://a:8101", "qwen-0.5b"
        )
        await registry.update(worker.id, WorkerUpdate(enabled=False))
        with pytest.raises(NoAvailableWorkerError):
            await registry.assign("qwen-0.5b")

    asyncio.run(scenario())


async def _healthy(
    registry: WorkerRegistry,
    name: str,
    base_url: str,
    model: str,
) -> Worker:
    worker = await registry.create(
        WorkerCreate(
            name=name,
            base_url=base_url,
            metadata={"model": model},
        )
    )
    updated = await registry.set_health_status(
        worker.id,
        HealthStatus.HEALTHY,
        expected_base_url=str(worker.base_url),
    )
    assert updated is not None
    return updated
