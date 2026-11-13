from tokenflow.worker_manager.models import HealthStatus, Worker


class NoAvailableWorkerError(LookupError):
    def __init__(self, model: str) -> None:
        super().__init__(
            f"No enabled, healthy worker is serving model '{model}'"
        )
        self.model = model


def choose_worker(workers: list[tuple[Worker, int]], model: str) -> Worker:
    """Pick the least-busy worker that is ready to serve this model.

    A worker is ready when it is enabled. It must also be healthy, and its
    declared model must match the request. Different model names, such as a
    0.5B deployment and a 14B deployment, stay in separate routing pools.
    Equal loads keep the earlier worker in the given order.
    """
    eligible = [
        (worker, active_requests)
        for worker, active_requests in workers
        if worker.enabled
        and worker.health_status is HealthStatus.HEALTHY
        and worker.metadata.get("model") == model
    ]
    if not eligible:
        raise NoAvailableWorkerError(model)
    worker, _active_requests = min(eligible, key=lambda item: item[1])
    return worker
