import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tokenflow.platform import controller as module
from tokenflow.platform.controller import Controller
from tokenflow.provisioner.base import PoolSpec, Resource
from tokenflow.scheduler.selection import NoAvailableWorkerError, WorkersAtCapacityError
from tokenflow.worker_manager import registry as registry_module
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate, WorkerTelemetry, WorkerUpdate
from tokenflow.worker_manager.registry import WorkerRegistry


class Inventory:
    """Provider accepting launch before the caller receives its reply."""

    def __init__(self):
        self.items = {}
        self.launches = []
        self.terminated = []
        self.lost_reply = False
        self.offline = False

    async def discover(self, pool):
        if self.offline:
            raise ConnectionError("inventory unavailable")
        return list(self.items.get(pool.name, {}).values())

    async def launch(self, pool, key):
        self.launches.append((pool.name, key))
        resource = Resource(key, f"job-{len(self.launches)}", "pending")
        self.items.setdefault(pool.name, {})[key] = resource
        if self.lost_reply:
            raise TimeoutError("reply lost")
        return resource

    async def terminate(self, pool, resource):
        self.terminated.append(resource.key)
        self.items[pool.name][resource.key] = Resource(resource.key, resource.external_id, "terminal")

    def boot(self, name, key):
        pending = self.items[name][key]
        self.items[name][key] = Resource(key, pending.external_id, "running", "http://gpu:18000")


def spec(name="big", **changes):
    fields = dict(
        name=name, model="qwen", checkpoint="qwen-checkpoint", image="engine", gpu_type="A6000",
        provider="slurm", min_replicas=1, max_replicas=3, capacity=2,
    )
    return PoolSpec(**(fields | changes))


def create(path: Path, inventory: Inventory, *pools: PoolSpec):
    return Controller(list(pools), {p.provider: inventory for p in pools}, WorkerRegistry(), path,
                      interval_seconds=1, idle_seconds=10, startup_timeout_seconds=20)


def test_adoption_after_lost_reply_and_restart_drains_before_terminating(tmp_path):
    async def scenario():
        path = tmp_path / "state.db"
        inventory = Inventory()
        inventory.lost_reply = True
        first = create(path, inventory, spec())
        await first.reconcile()
        assert "reply lost" in first.status()[0]["last_error"]
        _, key = inventory.launches[0]
        first.close()

        inventory.boot("big", key)
        controller = create(path, inventory, spec())
        await controller.reconcile()
        [worker] = await controller.registry.list()
        assert worker.metadata["model"] == "qwen"
        assert worker.metadata["gpu_metrics_port"] == 18001
        assert worker.health_status == HealthStatus.UNKNOWN
        assert controller.slurm_key_for_url(httpx.URL("http://gpu:18001/metrics")) == key
        assert controller.slurm_key_for_url(httpx.URL("https://gpu:18000/health")) is None
        await controller.registry.set_health_status(worker.id, HealthStatus.HEALTHY,
                                                    expected_base_url=str(worker.base_url))
        await controller.registry.assign("qwen")
        controller.set_limits("big", 0, 0)
        await controller.reconcile()
        assert not (await controller.registry.get(worker.id)).enabled
        with pytest.raises(NoAvailableWorkerError):
            await controller.registry.assign("qwen")
        assert inventory.terminated == []
        await controller.registry.release(worker.id)
        await controller.reconcile()
        assert inventory.terminated == [key]
        await controller.reconcile()
        assert not await controller.registry.list()
        assert not controller.status()[0]["resources"]
        assert controller.slurm_key_for_url(httpx.URL("http://gpu:18000/health")) is None
        assert len(inventory.launches) == 1
        controller.close()
        restored = create(path, inventory, spec())
        assert restored.status()[0]["max_replicas"] == 0
        await restored.reconcile()
        assert len(inventory.launches) == 1
        restored.close()

    asyncio.run(scenario())


def test_zero_to_one_selects_one_hardware_pool_and_wrong_model_stays_at_zero(tmp_path):
    async def scenario():
        inventory = Inventory()
        pools = [
            spec("slow", min_replicas=0, service_time_seconds=2),
            spec("fast", min_replicas=0, service_time_seconds=1),
        ]
        controller = create(tmp_path / "state.db", inventory, *pools)
        with pytest.raises(NoAvailableWorkerError):
            await controller.registry.assign("typo")
        await controller.reconcile()
        assert not inventory.launches
        with pytest.raises(NoAvailableWorkerError):
            await controller.registry.assign("qwen")
        await controller.reconcile()
        assert [name for name, _ in inventory.launches] == ["fast"]
        inventory.offline = True
        await controller.reconcile()
        assert not inventory.terminated
        assert len(inventory.launches) == 1
        assert controller.status()[1]["last_error"] == "inventory unavailable"
        controller.close()

    asyncio.run(scenario())


def test_queued_slurm_job_keeps_priority_but_failed_startup_is_replaced(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec())
        await controller.reconcile()
        _, key = inventory.launches[0]
        now[0] += 3600
        await controller.reconcile()
        assert not inventory.terminated
        inventory.boot("big", key)
        await controller.reconcile()
        now[0] += 21
        await controller.reconcile()
        assert inventory.terminated == [key]
        assert "readiness deadline" in controller.status()[0]["last_error"]
        assert len(inventory.launches) == 1
        await controller.reconcile()
        assert not controller.status()[0]["resources"]
        now[0] += 301
        await controller.reconcile()
        assert len(inventory.launches) == 2
        assert inventory.launches[1][1] != key
        controller.close()

    asyncio.run(scenario())


def test_single_cold_arrival_through_slurm_queue_and_model_startup(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
        monkeypatch.setattr(registry_module, "monotonic", lambda: now[0])
        provider = Inventory()
        registry = WorkerRegistry()
        controller = Controller([spec(min_replicas=0)], {"slurm": provider}, registry, tmp_path / "state.db")
        try:
            await controller.reconcile()
            with pytest.raises(NoAvailableWorkerError):
                await registry.assign("qwen")  # The only arrival before readiness.
            provider.offline = True
            now[0] += 10
            await controller.reconcile()
            provider.offline = False
            now[0] += 35  # Beyond the normal 20s demand window.
            await controller.reconcile()
            assert len(provider.launches) == 1
            _, key = provider.launches[0]
            now[0] += 3600
            await controller.reconcile()
            now[0] += 3600
            await controller.reconcile()
            assert provider.terminated == []
            # Slurm has assigned the node. The engine publishes its endpoint
            # after the image and model load, then the monitor probes it.
            resource = provider.items["big"][key]
            provider.items["big"][key] = Resource(key, resource.external_id, "running")
            await controller.reconcile()
            now[0] += 900
            provider.boot("big", key)
            await controller.reconcile()
            [worker] = await registry.list()
            assert worker.health_status == HealthStatus.UNKNOWN
            await registry.set_health_status(worker.id, HealthStatus.HEALTHY, expected_base_url=str(worker.base_url))
            await controller.reconcile()
            now[0] += 299
            await controller.reconcile()
            assert (await registry.assign("qwen")).id == worker.id
            assert len(provider.launches) == 1
            assert provider.terminated == []
            await registry.release(worker.id)
            await controller.reconcile()
            now[0] += 30
            await controller.reconcile()
            now[0] += 301
            await controller.reconcile()
            assert provider.terminated == [key]
        finally:
            controller.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("pause_before_recovery,delay", [(False, 360), (True, 45)])
def test_expired_or_paused_cold_arrival_does_not_launch(tmp_path, monkeypatch, pause_before_recovery, delay):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(registry_module, "monotonic", lambda: now[0])
        provider = Inventory()
        controller = create(tmp_path / "state.db", provider, spec(min_replicas=0))
        try:
            with pytest.raises(NoAvailableWorkerError):
                await controller.registry.assign("qwen")
            if pause_before_recovery:
                controller.set_limits("big", 0, 0)
                with pytest.raises(NoAvailableWorkerError):
                    await controller.registry.assign("qwen")
                controller.set_limits("big", 0, 1)
            now[0] += delay
            await controller.reconcile()
            assert provider.launches == []
        finally:
            controller.close()

    asyncio.run(scenario())


def test_load_drives_scale_up_and_cooldown_prevents_immediate_scale_down(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec(min_replicas=0, capacity=1))
        with pytest.raises(NoAvailableWorkerError):
            await controller.registry.assign("qwen")
        await controller.reconcile()
        _, key = inventory.launches[0]
        inventory.boot("big", key)
        await controller.reconcile()
        [worker] = await controller.registry.list()
        await controller.registry.set_health_status(worker.id, HealthStatus.HEALTHY,
                                                    expected_base_url=str(worker.base_url))
        await controller.registry.assign("qwen")
        with pytest.raises(WorkersAtCapacityError):
            await controller.registry.assign("qwen")
        await controller.reconcile()
        assert len(inventory.launches) == 2
        await controller.registry.release(worker.id)
        monkeypatch.setattr(controller.registry, "demand_since", lambda model, seconds: 0)
        now[0] += 3
        await controller.reconcile()
        assert not inventory.terminated
        controller.close()

    asyncio.run(scenario())


def test_reconfiguration_preserves_ownership_of_existing_jobs(tmp_path):
    async def scenario():
        inventory = Inventory()
        path = tmp_path / "state.db"
        first = create(path, inventory, spec())
        await first.reconcile()
        with pytest.raises(ValueError):
            first.set_limits("big", 0, 4)
        first.close()
        with pytest.raises(ValueError, match="Drain pool"):
            create(path, inventory, spec(checkpoint="other"))
        with pytest.raises(ValueError, match="Drain pool"):
            create(path, inventory, spec("renamed"))
        with pytest.raises(ValueError, match="Drain pool"):
            create(path, inventory, spec(capacity=42))

    asyncio.run(scenario())


def test_temporary_inventory_loss_recovers_same_worker_and_confirmed_termination_removes_it(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec())
        await controller.reconcile()
        _, key = inventory.launches[0]
        inventory.boot("big", key)
        await controller.reconcile()
        [worker] = await controller.registry.list()
        url = httpx.URL("http://gpu:18000")

        inventory.offline = True
        await controller.reconcile()
        assert controller.slurm_key_for_url(url) is None
        assert not (await controller.registry.get(worker.id)).enabled

        now[0] += 40
        inventory.offline = False
        running = inventory.items["big"].pop(key)
        await controller.reconcile()
        assert len(controller.status()[0]["resources"]) == 1
        inventory.items["big"][key] = running
        await controller.reconcile()
        assert (await controller.registry.get(worker.id)).enabled
        assert controller.slurm_key_for_url(url) == key
        assert len(inventory.launches) == 1
        inventory.items["big"][key] = Resource(key, running.external_id, "terminal")
        await controller.reconcile()
        assert not await controller.registry.list()
        assert controller.slurm_key_for_url(url) is None
        assert not controller.status()[0]["resources"]
        controller.close()

    asyncio.run(scenario())


def test_stopped_cloud_instance_hits_readiness_deadline(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec(provider="cloud"))
        await controller.reconcile()
        await controller.reconcile()
        now[0] += 21
        await controller.reconcile()
        assert inventory.terminated == [inventory.launches[0][1]]
        controller.close()

    asyncio.run(scenario())


def test_reassigned_private_ip_keeps_old_lease_until_drain(tmp_path):
    async def scenario():
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec(provider="cloud", max_replicas=1))
        await controller.reconcile()
        _, old = inventory.launches[0]
        inventory.boot("big", old)
        await controller.reconcile()
        [worker] = await controller.registry.list()
        await controller.registry.set_health_status(worker.id, HealthStatus.HEALTHY,
                                                    expected_base_url=str(worker.base_url))
        await controller.registry.assign("qwen")
        resource = inventory.items["big"].pop(old)
        inventory.items["big"]["replacement"] = Resource("replacement", "job-2", "running", resource.base_url)
        await controller.reconcile()
        assert not (await controller.registry.get(worker.id)).enabled
        assert (await controller.registry.get(worker.id)).active_requests == 1
        await controller.registry.release(worker.id)
        await controller.reconcile()
        [replacement] = await controller.registry.list()
        assert replacement.metadata["resource_key"] == "replacement"
        assert replacement.health_status == HealthStatus.UNKNOWN
        assert controller.owns_worker(replacement.id)
        assert not controller.owns_worker(worker.id)
        assert inventory.terminated == []
        controller.close()

    asyncio.run(scenario())


def test_reused_gateway_registry_recovers_its_managed_record(tmp_path):
    async def scenario():
        path = tmp_path / "state.db"
        inventory = Inventory()
        controller = create(path, inventory, spec())
        await controller.reconcile()
        _, key = inventory.launches[0]
        inventory.boot("big", key)
        await controller.reconcile()
        [original] = await controller.registry.list()
        controller.close()
        restarted = Controller([spec()], {"slurm": inventory}, controller.registry, path)
        await restarted.reconcile()
        [reattached] = await restarted.registry.list()
        assert reattached.id == original.id
        assert restarted.owns_worker(reattached.id)
        assert len(inventory.launches) == 1
        restarted.close()

    asyncio.run(scenario())


def test_paused_pool_stays_paused_after_checkpoint_change(tmp_path):
    async def scenario():
        path = tmp_path / "state.db"
        inventory = Inventory()
        controller = create(path, inventory, spec())
        controller.set_limits("big", 0, 0)
        controller.close()
        changed = create(path, inventory, spec(checkpoint="updated-checkpoint"))
        await changed.reconcile()
        assert changed.status()[0]["max_replicas"] == 0
        assert not inventory.launches
        changed.close()

    asyncio.run(scenario())


def test_tighter_catalog_ceiling_clamps_saved_override_and_reconciles_existing_jobs(tmp_path):
    async def scenario():
        path = tmp_path / "state.db"
        inventory = Inventory()
        controller = create(path, inventory, spec())
        controller.set_limits("big", 2, 3)
        await controller.reconcile()
        await controller.reconcile()
        assert len(inventory.launches) == 2
        controller.close()
        tighter = create(path, inventory, spec(max_replicas=1))
        await tighter.reconcile()
        assert (tighter.status()[0]["min_replicas"], tighter.status()[0]["max_replicas"]) == (1, 1)
        assert len(inventory.terminated) == 1
        assert len(inventory.launches) == 2
        tighter.close()

    asyncio.run(scenario())


def test_lost_submission_reply_is_not_retried_after_demand_expires(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
        inventory = Inventory()
        # A transport failure before the request reaches the scheduler and a
        # lost reply after acceptance look identical at the call site.
        async def fail_before_acceptance(pool, key):
            inventory.launches.append((pool.name, key))
            raise TimeoutError("submit outcome unknown")

        monkeypatch.setattr(inventory, "launch", fail_before_acceptance)
        controller = create(tmp_path / "state.db", inventory, spec(min_replicas=0))
        with pytest.raises(NoAvailableWorkerError):
            await controller.registry.assign("qwen")
        await controller.reconcile()
        assert len(inventory.launches) == 1
        monkeypatch.setattr(controller.registry, "demand_since", lambda model, seconds: 0)
        now[0] += 6  # First launch backoff expired; still in scale-down cooldown.
        await controller.reconcile()
        now[0] += 60
        await controller.reconcile()
        assert len(inventory.launches) == 1
        controller.close()

    asyncio.run(scenario())


def test_pause_while_terminating_prevents_launch_from_old_plan(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec(min_replicas=2, max_replicas=2))
        await controller.reconcile()  # One job submitted, second slot still empty.

        async def slow_termination(pool, resource):
            inventory.terminated.append(resource.key)  # Inventory still shows the job.
            controller.set_limits(pool.name, 0, 0)  # Operator update during the RPC.

        monkeypatch.setattr(inventory, "terminate", slow_termination)
        controller.set_limits("big", 0, 0)
        await controller.reconcile()
        controller.set_limits("big", 2, 2)
        now[0] += 31
        await controller.reconcile()
        assert len(inventory.terminated) == 2
        assert len(inventory.launches) == 1
        assert controller.status()[0]["desired_replicas"] == 0
        controller.close()

    asyncio.run(scenario())


def test_manual_capacity_covers_gateway_demand_until_external_load_exhausts_it(tmp_path):
    async def scenario():
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory,
                            spec("hpc", min_replicas=0, service_time_seconds=2),
                            spec("aws", provider="cloud", min_replicas=0, capacity=1))
        registry = controller.registry
        manual = await registry.create(WorkerCreate(
            name="manual", base_url="http://manual:8000",
            metadata={"model": "qwen", "capacity": 2, "pool": "aws"},
        ))
        url = str(manual.base_url)
        await registry.set_health_status(manual.id, HealthStatus.HEALTHY, expected_base_url=url)
        first = await registry.assign("qwen")
        # The engine's running request includes TokenFlow's live lease.
        await registry.set_telemetry(manual.id, WorkerTelemetry(observed_at=datetime.now(UTC), running=1),
                                     expected_base_url=url)
        second = await registry.assign("qwen")
        await controller.reconcile()
        assert not inventory.launches
        await registry.release(first.id)
        await registry.release(second.id)

        await registry.set_telemetry(manual.id, WorkerTelemetry(observed_at=datetime.now(UTC), running=2, waiting=500),
                                     expected_base_url=url)
        with pytest.raises(WorkersAtCapacityError):
            await registry.assign("qwen")
        await controller.reconcile()
        await controller.reconcile()
        # The real gateway peak of two fits one HPC replica. The direct
        # backlog of 500 stays on the manual worker that received it.
        assert [name for name, _ in inventory.launches] == ["hpc"]
        controller.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("sample_age,expected_extra", [(0, 1), (120, 0)])
def test_saturated_managed_engine_can_spill_into_another_provider(tmp_path, sample_age, expected_extra):
    async def scenario():
        slurm, cloud = Inventory(), Inventory()
        registry = WorkerRegistry()
        controller = Controller(
            [spec("hpc", min_replicas=1, max_replicas=1, capacity=2),
             spec("aws", provider="cloud", min_replicas=0, max_replicas=4, capacity=1)],
            {"slurm": slurm, "cloud": cloud}, registry, tmp_path / "state.db",
        )
        await controller.reconcile()
        _, key = slurm.launches[0]
        slurm.boot("hpc", key)
        await controller.reconcile()
        [worker] = await registry.list()
        url = str(worker.base_url)
        await registry.set_health_status(worker.id, HealthStatus.HEALTHY, expected_base_url=url)
        sample = WorkerTelemetry(observed_at=datetime.now(UTC) - timedelta(seconds=sample_age), running=2, waiting=500)
        await registry.set_telemetry(worker.id, sample, expected_base_url=url)
        await controller.reconcile()
        assert not cloud.launches  # Direct engine traffic alone triggers no paid expansion.
        if expected_extra:
            with pytest.raises(WorkersAtCapacityError):
                await registry.assign("qwen")
        else:
            assert (await registry.assign("qwen")).id == worker.id
        for _ in range(4):
            await controller.reconcile()
        assert len(slurm.launches) == 1
        assert len(cloud.launches) == expected_extra
        controller.close()

    asyncio.run(scenario())


def test_missing_manual_capacity_does_not_block_scale_from_zero(tmp_path):
    async def scenario():
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec(min_replicas=0))
        registry = controller.registry
        for name, model, enabled, health in [
            ("other", "different", True, HealthStatus.HEALTHY),
            ("disabled", "qwen", False, HealthStatus.HEALTHY),
            ("unhealthy", "qwen", True, HealthStatus.UNHEALTHY),
        ]:
            worker = await registry.create(WorkerCreate(
                name=name, base_url=f"http://{name}:8000", metadata={"model": model},
            ))
            await registry.set_health_status(worker.id, health, expected_base_url=str(worker.base_url))
            if not enabled:
                await registry.update(worker.id, WorkerUpdate(enabled=False))
        with pytest.raises(NoAvailableWorkerError):
            await registry.assign("qwen")
        await controller.reconcile()
        assert len(inventory.launches) == 1
        controller.close()

    asyncio.run(scenario())


def test_idle_downscale_keeps_externally_busy_worker_until_operator_pauses_pool(tmp_path, monkeypatch):
    async def scenario():
        now = [1_000.0]
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
        inventory = Inventory()
        controller = create(tmp_path / "state.db", inventory, spec(min_replicas=2, max_replicas=2))
        await controller.reconcile()
        await controller.reconcile()
        keys = [key for _, key in inventory.launches]
        for index, key in enumerate(keys):
            item = inventory.items["big"][key]
            inventory.items["big"][key] = Resource(key, item.external_id, "running", f"http://gpu{index}:18000")
        await controller.reconcile()
        for worker in await controller.registry.list():
            await controller.registry.set_health_status(worker.id, HealthStatus.HEALTHY,
                                                        expected_base_url=str(worker.base_url))
            if worker.metadata["resource_key"] == keys[1]:
                await controller.registry.set_telemetry(worker.id, WorkerTelemetry(observed_at=datetime.now(UTC), running=1),
                                                        expected_base_url=str(worker.base_url))
        controller.set_limits("big", 0, 2)
        await controller.reconcile()
        now[0] += 11
        await controller.reconcile()
        assert inventory.terminated == [keys[0]]
        controller.set_limits("big", 0, 0)
        await controller.reconcile()
        assert inventory.terminated == keys
        assert len(inventory.launches) == 2
        controller.close()

    asyncio.run(scenario())
