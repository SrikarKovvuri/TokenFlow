from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from uuid import UUID, uuid4

from tokenflow.scheduler.selection import NoAvailableWorkerError, choose_worker
from tokenflow.worker_manager.models import HealthStatus, Worker, WorkerCreate, WorkerTelemetry, WorkerUpdate

DEMAND_HISTORY_SECONDS = 300


@dataclass
class TelemetryProbe:
    worker: Worker
    started_at: datetime
    gateway_overlap: int


class WorkerNotFoundError(LookupError):
    def __init__(self, worker_id: UUID) -> None:
        super().__init__(f"Worker '{worker_id}' was not found")


class DuplicateWorkerError(ValueError):
    def __init__(self, base_url: str) -> None:
        super().__init__(f"A worker with URL '{base_url}' is already registered")


class WorkerRegistry:
    """Process-local memory of registered inference workers.

    ``_workers`` is the registration record. ``_inflight`` is the live count of
    requests assigned to each worker and not yet finished. The count stays in
    its own map so a health update cannot overwrite it.
    """

    def __init__(self, *, telemetry_age_seconds: float = 30) -> None:
        self._workers: dict[UUID, Worker] = {}
        self._inflight: dict[UUID, int] = {}
        self._prior_endpoint: dict[UUID, int] = {}
        self._probes: dict[UUID, TelemetryProbe] = {}
        self._telemetry_age_seconds = telemetry_age_seconds
        self._demand: dict[str, deque[tuple[int, int]]] = {}
        self._cold_arrivals: dict[str, float] = {}

    def track_demand(self, models: Iterable[str]) -> None:
        """Track only configured models so arbitrary request names cannot grow state."""
        self._demand = {
            model: self._demand.get(model, deque(maxlen=DEMAND_HISTORY_SECONDS + 1)) for model in models
        }
        self._cold_arrivals = {m: when for m, when in self._cold_arrivals.items() if m in self._demand}

    def demand_since(self, model: str, seconds: float) -> int:
        """Peak offered concurrency in a rolling window of up to five minutes."""
        cutoff = int(monotonic() - min(DEMAND_HISTORY_SECONDS, seconds))
        return max(
            (peak for when, peak in self._demand.get(model, ()) if when >= cutoff),
            default=0,
        )

    def take_cold_arrival(self, model: str) -> bool:
        """Allow one cold-start decision when a slow inventory poll misses the rolling peak."""
        when = self._cold_arrivals.pop(model, None)
        return when is not None and monotonic() - when <= DEMAND_HISTORY_SECONDS

    def capacity_for_gateway(self, worker: Worker) -> int:
        """Slots available for both current and incoming gateway reservations."""
        sample = worker.fresh_telemetry(datetime.now(UTC), self._telemetry_age_seconds)
        return worker.gateway_capacity(sample)

    def _record_arrival(self, model: str) -> None:
        history = self._demand.get(model)
        if history is None:
            return
        demand = 1 + sum(
            self._inflight.get(worker.id, 0)
            for worker in self._workers.values() if worker.metadata.get("model") == model
        )
        second = int(monotonic())
        if history and history[-1][0] == second:
            demand = max(demand, history.pop()[1])
        history.append((second, demand))

    async def create(self, worker_data: WorkerCreate, *, worker_id: UUID | None = None) -> Worker:
        self._ensure_url_available(str(worker_data.base_url))
        worker_id = worker_id or uuid4()
        if worker_id in self._workers:
            raise ValueError(f"Worker ID '{worker_id}' is already registered")
        now = datetime.now(UTC)
        worker = Worker(
            **worker_data.model_dump(),
            id=worker_id,
            registered_at=now,
            updated_at=now,
        )
        self._workers[worker.id] = worker
        return worker

    async def list(self) -> list[Worker]:
        return [self._with_load(worker) for worker in self._workers.values()]

    async def get(self, worker_id: UUID) -> Worker:
        return self._with_load(self._get(worker_id))

    async def assign(self, model: str, *, exclude: set[UUID] | None = None) -> Worker:
        """Select and reserve a worker before any network wait."""
        if not exclude:
            self._record_arrival(model)
        candidates = [(candidate, self._inflight.get(candidate.id, 0))
                      for candidate in self._workers.values() if not exclude or candidate.id not in exclude]
        try:
            worker = choose_worker(candidates, model, telemetry_age_seconds=self._telemetry_age_seconds)
        except NoAvailableWorkerError:
            if not exclude and model in self._demand:
                self._cold_arrivals[model] = monotonic()
            raise
        self._inflight[worker.id] = self._inflight.get(worker.id, 0) + 1
        return self._with_load(worker)

    async def release(self, worker_id: UUID, *, registered_at: datetime | None = None) -> None:
        """Drop one in-flight reservation, including after cancellation."""
        worker = self._workers.get(worker_id)
        if registered_at is not None and (worker is None or registered_at != worker.registered_at):
            return  # A managed allocation can re-register with the same ID.
        current = self._inflight.get(worker_id, 0)
        # The engine does not identify individual gateway requests in its gauges.
        # A release during a scrape might finish one of the original leases.
        probe = self._probes.get(worker_id)
        if current and probe and probe.gateway_overlap:
            probe.gateway_overlap -= 1
        if current <= 1:
            self._inflight.pop(worker_id, None)
            self._prior_endpoint.pop(worker_id, None)
        else:
            self._inflight[worker_id] = current - 1
            if worker_id in self._prior_endpoint:
                self._prior_endpoint[worker_id] = min(self._prior_endpoint[worker_id], current - 1)

    async def update(
        self,
        worker_id: UUID,
        worker_data: WorkerUpdate,
    ) -> Worker:
        worker = self._get(worker_id)
        changes = worker_data.model_dump(exclude_unset=True)

        if "base_url" in changes:
            self._ensure_url_available(
                str(changes["base_url"]),
                excluding=worker_id,
            )

        updated_data = worker.model_dump()
        updated_data.update(changes)
        # A new address or a worker being turned back on has not been checked yet.
        if "base_url" in changes or (
            changes.get("enabled") is True and not worker.enabled
        ):
            updated_data["health_status"] = HealthStatus.UNKNOWN
            updated_data["telemetry"] = None
            self._reset_endpoint(worker_id)
        if "metadata" in changes and changes["metadata"] != worker.metadata:
            updated_data["telemetry"] = None
        updated_data["updated_at"] = datetime.now(UTC)
        updated_worker = Worker.model_validate(updated_data)
        self._workers[worker_id] = updated_worker
        return self._with_load(updated_worker)

    async def set_health_status(
        self,
        worker_id: UUID,
        health_status: HealthStatus,
        *,
        expected_base_url: str,
    ) -> Worker | None:
        """Record one health check if that check is still about this worker.

        A slow check can finish after the worker was deleted or its URL changed.
        Those results are discarded instead of marking the new record.
        """
        worker = self._workers.get(worker_id)
        if worker is None or not worker.enabled or str(worker.base_url) != expected_base_url:
            return None
        if worker.health_status is health_status:
            return worker

        updated_data = worker.model_dump()
        updated_data["health_status"] = health_status
        if health_status is not HealthStatus.HEALTHY:
            updated_data["telemetry"] = None
            self._reset_endpoint(worker_id)
        updated_data["updated_at"] = datetime.now(UTC)
        updated_worker = Worker.model_validate(updated_data)
        self._workers[worker_id] = updated_worker
        return updated_worker

    def _reset_endpoint(self, worker_id: UUID) -> None:
        self._probes.pop(worker_id, None)
        if self._inflight.get(worker_id):
            self._prior_endpoint[worker_id] = self._inflight[worker_id]

    def _possible_overlap(self, worker_id: UUID) -> int:
        # Leases still draining from an earlier endpoint may never appear on
        # this engine. Their count is an upper bound until completions arrive.
        return self._inflight.get(worker_id, 0) - self._prior_endpoint.get(worker_id, 0)

    def begin_telemetry_probe(self, worker_id: UUID) -> TelemetryProbe:
        probe = TelemetryProbe(self._get(worker_id), datetime.now(UTC), self._possible_overlap(worker_id))
        self._probes[worker_id] = probe
        return probe

    async def set_telemetry(
        self, worker_id: UUID, telemetry: WorkerTelemetry | None, *, expected_base_url: str,
        probe: TelemetryProbe | None = None,
    ) -> None:
        worker = self._workers.get(worker_id)
        if probe is not None and self._probes.get(worker_id) is not probe:
            return  # A newer probe or registration has superseded this one.
        if (worker is None or not worker.enabled or str(worker.base_url) != expected_base_url
                or worker.health_status != HealthStatus.HEALTHY or (probe and probe.worker is not worker)):
            return
        self._probes.pop(worker_id, None)
        if telemetry is not None:
            overlap = probe.gateway_overlap if probe else self._possible_overlap(worker_id)
            telemetry = telemetry.model_copy(update={
                "gateway_overlap": min(telemetry.requests, overlap),
                "observed_at": min(telemetry.observed_at, probe.started_at) if probe else telemetry.observed_at,
            })
        self._workers[worker_id] = worker.model_copy(update={"telemetry": telemetry})

    async def delete(self, worker_id: UUID) -> None:
        self._get(worker_id)
        self.discard(worker_id)

    def discard(self, worker_id: UUID) -> None:
        self._workers.pop(worker_id, None)
        self._inflight.pop(worker_id, None)
        self._prior_endpoint.pop(worker_id, None)
        self._probes.pop(worker_id, None)

    def _with_load(self, worker: Worker) -> Worker:
        # Copy the live count onto the response without changing the stored record.
        return worker.model_copy(
            update={"active_requests": self._inflight.get(worker.id, 0)}
        )

    def _get(self, worker_id: UUID) -> Worker:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise WorkerNotFoundError(worker_id) from exc

    def _ensure_url_available(
        self,
        base_url: str,
        *,
        excluding: UUID | None = None,
    ) -> None:
        if any(
            str(worker.base_url) == base_url and worker.id != excluding
            for worker in self._workers.values()
        ):
            raise DuplicateWorkerError(base_url)
