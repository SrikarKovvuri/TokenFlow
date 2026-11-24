"""Reconcile provider inventory, worker health and per-model demand in one gateway."""

import asyncio
import logging
import math
import time
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from tokenflow.platform.state import Allocation, State
from tokenflow.provisioner.base import PoolSpec, Provisioner, Resource
from tokenflow.worker_manager.models import HealthStatus, Worker, WorkerCreate, WorkerUpdate
from tokenflow.worker_manager.registry import DuplicateWorkerError, WorkerRegistry

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self, pools: list[PoolSpec], providers: dict[str, Provisioner],
        registry: WorkerRegistry, state_path: Path, *, interval_seconds: float = 10,
        idle_seconds: float = 300, startup_timeout_seconds: float = 1800,
    ):
        self.pools = {pool.name: pool for pool in pools}
        self.providers = providers
        self.registry = registry
        self.interval = interval_seconds
        self.idle = idle_seconds
        self.startup_timeout = startup_timeout_seconds
        self.state = State(state_path, pools)
        self._lock = asyncio.Lock()
        self._workers = self.state.registrations()
        self._origins: dict[str, dict[tuple, str]] = {}
        self._models = {pool.model for pool in pools}
        self._missing_since: dict[tuple[str, str], float] = {}
        self._down_since: dict[str, float] = {}
        self._desired: dict[str, int] = {}
        self._discovery_errors: dict[str, str] = {}
        registry.track_demand(self._models)

    def close(self) -> None:
        # Allocations outlive a gateway restart. On startup, discovery reattaches them.
        self.state.close()

    def slurm_key_for_url(self, url: httpx.URL) -> str | None:
        origin = (url.scheme, url.host, url.port or (443 if url.scheme == "https" else 80))
        matches = {origins[origin] for origins in self._origins.values() if origin in origins}
        return matches.pop() if len(matches) == 1 else None

    def owns_worker(self, worker_id: UUID) -> bool:
        return worker_id in self._workers.values()

    def set_limits(self, pool_name: str, minimum: int, maximum: int) -> None:
        pool = self.pools[pool_name]
        if not 0 <= minimum <= maximum <= pool.max_replicas:
            raise ValueError(f"Expected 0 <= minimum <= maximum <= {pool.max_replicas}")
        self.state.limits(pool_name, minimum, maximum)

    def _limits(self, pool: PoolSpec) -> tuple[int, int]:
        row = self.state.policy(pool.name)
        if row["maximum"] is None:
            return pool.min_replicas, pool.max_replicas
        return row["minimum"], row["maximum"]

    def status(self) -> list[dict]:
        result = []
        for pool in self.pools.values():
            minimum, maximum = self._limits(pool)
            policy = self.state.policy(pool.name)
            result.append({
                "name": pool.name, "provider": pool.provider, "model": pool.model,
                "min_replicas": minimum, "max_replicas": maximum,
                "desired_replicas": min(maximum, max(minimum, self._desired.get(pool.name, minimum))),
                "last_error": self._discovery_errors.get(pool.name) or policy["error"],
                "retry_at": policy["retry_at"] or None,
                "resources": [
                    {"key": item.key, "external_id": item.external_id,
                     "phase": "draining" if item.retiring else item.phase,
                     "created": item.created, "last_seen": item.last_seen,
                     "worker_id": self._workers.get(item.key)}
                    for item in self.state.allocations(pool.name)
                ],
            })
        return result

    async def run(self) -> None:
        while True:
            try:
                await self.reconcile()
            except Exception:
                logger.exception("Platform reconciliation failed")
            await asyncio.sleep(self.interval)

    async def reconcile(self) -> None:
        async with self._lock:
            available = []
            for pool in self.pools.values():
                try:
                    await self._discover(pool)
                except Exception as exc:
                    # An unavailable provider is not evidence that its workers disappeared.
                    # Withhold new assignments and credentials until ownership is
                    # confirmed again. Keep the durable allocation and active leases.
                    self._origins.pop(pool.name, None)
                    for item in self.state.allocations(pool.name):
                        await self._disable(item.key)
                    self._discovery_errors[pool.name] = str(exc)
                    logger.warning("Discovery failed for %s: %s", pool.name, exc)
                else:
                    self._discovery_errors.pop(pool.name, None)
                    available.append(pool)

            workers = {w.id: w for w in await self.registry.list()}
            targets = self._targets(available, workers)
            for pool in available:
                try:
                    await self._resize(pool, targets[pool.name], workers)
                except Exception as exc:
                    self.state.failure(pool.name, time.time(), str(exc))
                    logger.exception("Resize failed for %s", pool.name)

    def _inventory_grace(self) -> float:
        return max(30, 3 * self.interval)

    async def _discover(self, pool: PoolSpec) -> None:
        now = time.time()
        resources = await self.providers[pool.provider].discover(pool)
        observed = {resource.key: resource for resource in resources}
        if len(observed) != len(resources):
            raise ValueError(f"Provider returned duplicate allocation keys for {pool.name}")
        origins = {}
        for resource in resources:
            if pool.provider == "slurm" and resource.phase == "running" and resource.base_url:
                url = httpx.URL(resource.base_url)
                port = url.port or (443 if url.scheme == "https" else 80)
                for service_port in (port, port + 1):
                    origin = (url.scheme, url.host, service_port)
                    if origin in origins:
                        raise ValueError(f"Two allocations claim the same service address in {pool.name}")
                    origins[origin] = resource.key
        self._origins[pool.name] = origins

        saved = {item.key: item for item in self.state.allocations(pool.name)}
        # A new EC2 instance can inherit a recently released private address.
        # Drop the obsolete registration before attaching its new owner. Existing
        # requests retain their lease until release; new assignments stay disabled.
        endpoints = [r for r in resources if r.phase == "running" and r.base_url]
        owners = {r.base_url: r.key for r in endpoints}
        if len(owners) != len(endpoints):
            raise ValueError(f"Provider returned duplicate running endpoints in {pool.name}")
        for item in saved.values():
            if item.base_url in owners and owners[item.base_url] != item.key:
                if item.key not in observed:
                    item.retiring = True
                    self.state.save(item)
                await self._disable(item.key)
                worker_id = self._workers.get(item.key)
                if worker_id and not (await self.registry.get(worker_id)).active_requests:
                    await self._unregister(item.key)

        for key, resource in observed.items():
            self._missing_since.pop((pool.name, key), None)
            item = saved.get(key)
            if resource.phase == "terminal":
                if item:
                    if not item.retiring:
                        self.state.failure(pool.name, now, f"Allocation {resource.external_id} ended")
                    await self._forget(item)
                continue
            if item is None:
                item = Allocation(key, pool.name, now, now)
            if item.external_id and item.external_id != resource.external_id:
                raise ValueError(f"Allocation {key} changed provider identity")
            item.external_id = resource.external_id
            item.last_seen = now
            item.phase = resource.phase
            item.base_url = str(resource.base_url) if resource.base_url else None
            await self._sync_worker(pool, item, now)
            worker_id = self._workers.get(item.key)
            item.worker_id = str(worker_id) if worker_id else None
            self.state.save(item)

        # Allow eventual consistency after launch/termination; withhold routing to
        # an unobserved endpoint while preserving its reservation and any live lease.
        for key, item in saved.items():
            if key in observed:
                continue
            await self._disable(key)
            if not item.external_id and not item.retiring:
                continue  # ambiguous launch: always retry the original durable key
            since = self._missing_since.setdefault((pool.name, key), now)
            if now - since >= self._inventory_grace():
                worker_id = self._workers.get(item.key)
                if worker_id and (await self.registry.get(worker_id)).active_requests:
                    continue
                if not item.retiring:
                    self.state.failure(pool.name, now, f"Allocation {item.external_id} disappeared")
                await self._forget(item)

    async def _sync_worker(self, pool: PoolSpec, item: Allocation, now: float) -> None:
        worker = None
        if item.phase == "running" and item.base_url and not item.retiring:
            url = httpx.URL(item.base_url)
            port = url.port or (443 if url.scheme == "https" else 80)
            metadata = {
                "pool": pool.name, "resource_key": item.key, "model": pool.model,
                "backend": pool.backend, "gpu_type": pool.gpu_type, "gpu_count": pool.gpu_count,
                "capacity": pool.capacity, "service_time_seconds": pool.service_time_seconds,
                "gpu_metrics_port": port + 1,
            }
            worker_id = self._workers.get(item.key)
            if worker_id:
                try:
                    worker = await self.registry.get(worker_id)
                except LookupError:
                    self._workers.pop(item.key, None)
            if worker is None:
                try:
                    worker = await self.registry.create(WorkerCreate(
                        name=f"{pool.name}-{item.key[:12]}", base_url=item.base_url, metadata=metadata,
                    ))
                except DuplicateWorkerError:
                    # A draining or manual record still owns this address. Other
                    # workers continue serving while this one awaits registration.
                    if item.unready_since is None:
                        logger.warning("Pool %s cannot register an occupied address: %s", pool.name, item.base_url)
                else:
                    self._workers[item.key] = worker.id
                    # On restart the health monitor has yet to poll this address.
                    item.unready_since = now
            else:
                changes = {}
                if str(worker.base_url).rstrip("/") != item.base_url.rstrip("/"):
                    changes["base_url"] = item.base_url
                if worker.metadata != metadata:
                    changes["metadata"] = metadata
                if not worker.enabled:
                    changes["enabled"] = True
                if changes:
                    worker = await self.registry.update(worker.id, WorkerUpdate(**changes))
                    if "enabled" in changes or "base_url" in changes:
                        # Rediscovery re-enables probes; an observation that could
                        # not run during a provider outage cannot justify eviction.
                        item.unready_since = now
        else:
            await self._disable(item.key)

        if item.retiring:
            return
        if item.phase == "pending" and pool.provider == "slurm":
            # In particular, an unallocated Slurm job can wait for hours. Restarting
            # it would discard its queue position without fixing any health issue.
            item.unready_since = None
            return
        if worker and worker.health_status == HealthStatus.HEALTHY:
            if not item.ever_ready:
                self.state.healthy(pool.name)
                self._down_since[pool.name] = now
            item.ever_ready = True
            item.unready_since = None
            return

        if item.unready_since is None:
            item.unready_since = now
        # Allow a cold checkpoint time to load; a previously healthy server gets a
        # shorter recovery window to ride out a transient health or network failure.
        grace = min(self.startup_timeout, 60) if item.ever_ready else self.startup_timeout
        if now - item.unready_since >= grace:
            item.retiring = True
            self.state.failure(pool.name, now, f"Allocation {item.external_id} exceeded its readiness deadline")
            await self._disable(item.key)

    async def _disable(self, key: str) -> None:
        worker_id = self._workers.get(key)
        if worker_id:
            try:
                worker = await self.registry.get(worker_id)
                if worker.enabled:
                    await self.registry.update(worker_id, WorkerUpdate(enabled=False))
            except LookupError:
                self._workers.pop(key, None)

    async def _unregister(self, key: str) -> None:
        worker_id = self._workers.pop(key, None)
        if worker_id:
            try:
                await self.registry.delete(worker_id)
            except LookupError:
                pass

    async def _forget(self, item: Allocation) -> None:
        self._missing_since.pop((item.pool, item.key), None)
        await self._unregister(item.key)
        self.state.remove(item.key)

    def _targets(self, pools: list[PoolSpec], workers: dict[UUID, Worker]) -> dict[str, int]:
        limits = {p.name: self._limits(p) for p in pools}
        targets: dict[str, int] = {}
        # Snapshot ready-worker capacity using the same freshness and engine-load
        # rules as admission. Pending allocations count toward future capacity.
        ready = {w.id: self.registry.capacity_for_gateway(w) for w in workers.values()
                 if w.enabled and w.health_status == HealthStatus.HEALTHY}
        managed_ids = set(self._workers.values())
        supply = {}
        for pool in pools:
            slots = [ready.get(self._workers.get(item.key), pool.capacity)
                     for item in self.state.allocations(pool.name) if not item.retiring]
            # Retain existing workers serving direct traffic during ordinary
            # downscaling. An operator's hard ceiling still takes precedence.
            busy = sorted((count for count in slots if count < pool.capacity), reverse=True)
            minimum, maximum = limits[pool.name]
            targets[pool.name] = min(maximum, max(minimum, len(busy)))
            supply[pool.name] = busy + [pool.capacity] * (len(slots) - len(busy))
        window = min(300, max(1, self.interval * 2))
        for model in self._models:
            candidates = [p for p in pools if p.model == model]
            serving = [w for w in workers.values() if w.metadata.get("model") == model]
            active = sum(w.active_requests for w in serving)
            manual_slots = sum(ready.get(w.id, 0) for w in serving if w.id not in managed_ids)
            shortfall = max(active, self.registry.demand_since(model, window)) - manual_slots
            for pool in candidates:
                floor = targets[pool.name]
                existing = supply[pool.name]
                shortfall -= sum(existing[:floor]) + max(0, floor - len(existing)) * pool.capacity

            # First use existing workers, including cold starts, before funding
            # extra replicas in order of configured per-GPU serving capacity.
            candidates.sort(key=lambda p: -p.capacity / (p.service_time_seconds * p.gpu_count))
            for pool in candidates:
                for slots in supply[pool.name][targets[pool.name]:limits[pool.name][1]]:
                    if shortfall <= 0:
                        break
                    targets[pool.name] += 1
                    shortfall -= slots
            for pool in candidates:
                if shortfall <= 0:
                    break
                if self.state.policy(pool.name)["retry_at"] > time.time():
                    continue
                room = limits[pool.name][1] - targets[pool.name]
                add = min(room, math.ceil(shortfall / pool.capacity))
                targets[pool.name] += add
                shortfall -= add * pool.capacity
        return targets

    async def _resize(self, pool: PoolSpec, target: int, workers: dict[UUID, Worker]) -> None:
        now = time.time()
        allocations = self.state.allocations(pool.name)
        live = [item for item in allocations if not item.retiring]
        limits = self._limits(pool)
        minimum, maximum = limits
        requested = min(maximum, max(minimum, target))
        target = requested
        if target < len(live) and len(live) <= maximum:
            since = self._down_since.setdefault(pool.name, now)
            if now - since < self.idle:
                target = len(live)
            elif maximum:
                # A scale-from-zero request should have a chance to become ready
                # before its short demand window is forgotten.
                target = max(target, sum(bool(item.external_id) and not item.ever_ready for item in live))
        else:
            self._down_since.pop(pool.name, None)
        target = min(maximum, target)
        self._desired[pool.name] = target
        if target < len(live):
            def drain_order(item: Allocation) -> tuple:
                worker = workers.get(self._workers.get(item.key))
                slots = pool.capacity
                if worker and worker.enabled and worker.health_status == HealthStatus.HEALTHY:
                    slots = self.registry.capacity_for_gateway(worker)
                # Keep directly used workers first. If the operator's ceiling
                # forces a choice among them, retain the usable slots we counted.
                return (slots < pool.capacity, slots, worker.active_requests if worker else 0,
                        item.ever_ready, -item.created)

            for item in sorted(live, key=drain_order)[:len(live) - target]:
                item.retiring = True
                self.state.save(item)

        for item in allocations:
            if item.retiring:
                await self._disable(item.key)
                worker_id = self._workers.get(item.key)
                if worker_id and (await self.registry.get(worker_id)).active_requests:
                    continue
                if (item.external_id and (pool.name, item.key) not in self._missing_since
                        and now >= item.terminate_after):
                    # A repeated terminate is harmless (e.g. controller restart
                    # after the provider accepted it but before SQLite committed).
                    item.terminate_after = now + self._inventory_grace()
                    self.state.save(item)
                    await self.providers[pool.provider].terminate(pool, Resource(
                        key=item.key, external_id=item.external_id, phase=item.phase, base_url=item.base_url,
                    ))

        # Termination awaits a provider RPC. The operator may have paused or
        # reduced the pool while that call was in flight.
        if self._limits(pool) != limits or self.state.policy(pool.name)["retry_at"] > now:
            return
        unsubmitted = next((a for a in allocations if not a.external_id and not a.retiring), None)
        if unsubmitted:
            # Retaining an already accepted cold start during the idle cooldown
            # must not initiate a paid retry after the original demand expires.
            confirmed = sum(bool(a.external_id) and not a.retiring for a in allocations)
            if confirmed < requested:
                await self._launch(pool, unsubmitted)
        elif len(allocations) < requested:
            # Only one new allocation per pool per poll. Terminating and unresolved
            # launches both occupy a slot until inventory confirms their outcome.
            item = Allocation(uuid4().hex, pool.name, now, now)
            self.state.save(item)
            await self._launch(pool, item)

    async def _launch(self, pool: PoolSpec, item: Allocation) -> None:
        resource = await self.providers[pool.provider].launch(pool, item.key)
        if resource.key != item.key:
            raise ValueError("Provider returned an unexpected allocation key")
        if resource.phase == "terminal":
            await self._forget(item)
            raise RuntimeError(f"Allocation {resource.external_id} ended during launch")
        item.external_id = resource.external_id
        item.phase = resource.phase
        item.last_seen = time.time()
        self.state.save(item)
