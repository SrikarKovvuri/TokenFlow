import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

from tokenflow.models import Worker, WorkerCreate, WorkerUpdate


class WorkerNotFoundError(LookupError):
    def __init__(self, worker_id: UUID) -> None:
        super().__init__(f"Worker '{worker_id}' was not found")


class DuplicateWorkerError(ValueError):
    def __init__(self, base_url: str) -> None:
        super().__init__(f"A worker with URL '{base_url}' is already registered")


class WorkerRegistry:
    """Process-local registry for configured inference workers."""

    def __init__(self) -> None:
        self._workers: dict[UUID, Worker] = {}
        self._lock = asyncio.Lock()

    async def create(self, worker_data: WorkerCreate) -> Worker:
        async with self._lock:
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
        async with self._lock:
            return list(self._workers.values())

    async def get(self, worker_id: UUID) -> Worker:
        async with self._lock:
            return self._get(worker_id)

    async def update(
        self,
        worker_id: UUID,
        worker_data: WorkerUpdate,
    ) -> Worker:
        async with self._lock:
            worker = self._get(worker_id)
            changes = worker_data.model_dump(exclude_unset=True)

            if "base_url" in changes:
                self._ensure_url_available(
                    str(changes["base_url"]),
                    excluding=worker_id,
                )

            updated_data = worker.model_dump()
            updated_data.update(changes)
            updated_data["updated_at"] = datetime.now(UTC)
            updated_worker = Worker.model_validate(updated_data)
            self._workers[worker_id] = updated_worker
            return updated_worker

    async def delete(self, worker_id: UUID) -> None:
        async with self._lock:
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
