from datetime import UTC, datetime
from uuid import UUID, uuid4

from tokenflow.scheduler.selection import choose_worker
from tokenflow.worker_manager.models import HealthStatus, Worker, WorkerCreate, WorkerUpdate


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

    def __init__(self) -> None:
        self._workers: dict[UUID, Worker] = {}
        self._inflight: dict[UUID, int] = {}

    async def create(self, worker_data: WorkerCreate) -> Worker:
        self._ensure_url_available(str(worker_data.base_url))
        now = datetime.now(UTC)
        worker = Worker(
            **worker_data.model_dump(),
            id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        self._workers[worker.id] = worker
        return worker

    async def list(self) -> list[Worker]:
        return [self._with_load(worker) for worker in self._workers.values()]

    async def get(self, worker_id: UUID) -> Worker:
        return self._with_load(self._get(worker_id))

    async def assign(self, model: str) -> Worker:
        """Reserve the least-busy eligible worker before any network wait.

        The count increases here, before the gateway contacts the worker, so
        the next request sees this reservation immediately.
        """
        worker = choose_worker(
            [
                (candidate, self._inflight.get(candidate.id, 0))
                for candidate in self._workers.values()
            ],
            model,
        )
        self._inflight[worker.id] = self._inflight.get(worker.id, 0) + 1
        return self._with_load(worker)

    async def release(self, worker_id: UUID) -> None:
        """Drop one in-flight reservation, including after cancellation."""
        current = self._inflight.get(worker_id, 0)
        if current <= 1:
            self._inflight.pop(worker_id, None)
        else:
            self._inflight[worker_id] = current - 1

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
        updated_data["updated_at"] = datetime.now(UTC)
        updated_worker = Worker.model_validate(updated_data)
        self._workers[worker_id] = updated_worker
        return updated_worker

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
        if worker is None or str(worker.base_url) != expected_base_url:
            return None
        if worker.health_status is health_status:
            return worker

        updated_data = worker.model_dump()
        updated_data["health_status"] = health_status
        updated_data["updated_at"] = datetime.now(UTC)
        updated_worker = Worker.model_validate(updated_data)
        self._workers[worker_id] = updated_worker
        return updated_worker

    async def delete(self, worker_id: UUID) -> None:
        self._get(worker_id)
        del self._workers[worker_id]
        self._inflight.pop(worker_id, None)

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
