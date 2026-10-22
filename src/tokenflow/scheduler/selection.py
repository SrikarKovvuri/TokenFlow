from tokenflow.worker_manager.models import HealthStatus, Worker
from tokenflow.worker_manager.registry import WorkerRegistry


class NoAvailableWorkerError(LookupError):
    def __init__(self) -> None:
        super().__init__("No enabled, healthy inference worker is available")


async def select_first_healthy(registry: WorkerRegistry) -> Worker:
    """Return the first registered worker that can receive traffic."""
    for worker in await registry.list():
        if worker.enabled and worker.health_status is HealthStatus.HEALTHY:
            return worker
    raise NoAvailableWorkerError()
