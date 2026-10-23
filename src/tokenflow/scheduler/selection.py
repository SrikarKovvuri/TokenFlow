from tokenflow.worker_manager.models import HealthStatus, Worker


class NoAvailableWorkerError(LookupError):
    def __init__(self) -> None:
        super().__init__("No enabled, healthy inference worker is available")


def select_first_healthy(workers: list[Worker]) -> Worker:
    """Choose from worker state already fetched from the worker manager."""
    for worker in workers:
        if worker.enabled and worker.health_status is HealthStatus.HEALTHY:
            return worker
    raise NoAvailableWorkerError()
