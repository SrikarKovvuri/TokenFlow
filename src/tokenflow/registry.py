from datetime import UTC, datetime
from uuid import UUID, uuid4

from tokenflow.models import HealthStatus, Worker, WorkerCreate, WorkerUpdate


class WorkerNotFoundError(LookupError):
    def __init__(self, worker_id: UUID) -> None:
        super().__init__(f"Worker '{worker_id}' was not found")


class DuplicateWorkerError(ValueError):
    def __init__(self, base_url: str) -> None:
        super().__init__(f"A worker with URL '{base_url}' is already registered")


class NoAvailableWorkerError(LookupError):
    def __init__(self) -> None:
        super().__init__("No enabled, healthy inference worker is available")


class WorkerRegistry:
    """Process-local registry for configured inference workers."""

    def __init__(self) -> None:
        self._workers: dict[UUID, Worker] = {}

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
        return list(self._workers.values())

    async def get(self, worker_id: UUID) -> Worker:
        return self._get(worker_id)

    async def get_available(self) -> Worker:
        """Return the first registered worker that can receive traffic."""
        for worker in self._workers.values():
            if (
                worker.enabled
                and worker.health_status is HealthStatus.HEALTHY
            ):
                return worker
        raise NoAvailableWorkerError()

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
    ) -> Worker | None:
        """Update observed health, ignoring workers removed during a check."""
        worker = self._workers.get(worker_id)
        if worker is None:
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
