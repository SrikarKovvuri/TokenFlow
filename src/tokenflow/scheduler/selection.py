from datetime import UTC, datetime

from tokenflow.worker_manager.models import HealthStatus, Worker


class NoAvailableWorkerError(LookupError):
    def __init__(self, model: str) -> None:
        super().__init__(f"No enabled, healthy worker is serving model '{model}'")
        self.model = model


class WorkersAtCapacityError(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(f"All available workers for model '{model}' are at capacity")


def choose_worker(
    workers: list[tuple[Worker, int]], model: str, *, telemetry_age_seconds: float = 30
) -> Worker:
    """Estimate delay from outstanding work, parallel capacity and service time.

    Account for gateway leases plus the engine's estimated external load.
    Equal delay favors more KV headroom, then a less busy GPU. Unknown gauges
    use a neutral midpoint. Identical candidates retain registration order.
    """
    now = datetime.now(UTC)
    candidates = []
    has_healthy = False
    for worker, active in workers:
        if (not worker.enabled or worker.health_status != HealthStatus.HEALTHY
                or worker.metadata.get("model") != model):
            continue
        has_healthy = True
        telemetry = worker.fresh_telemetry(now, telemetry_age_seconds)
        capacity = worker.capacity
        gateway_capacity = worker.gateway_capacity(telemetry)
        if active >= gateway_capacity:
            continue
        outstanding = active + (capacity - gateway_capacity)
        kv = gpu = 0.5
        if telemetry:
            if telemetry.kv_cache_usage is not None:
                kv = telemetry.kv_cache_usage
            if telemetry.gpu_utilization is not None:
                gpu = telemetry.gpu_utilization
        service_time = worker.metadata.get("service_time_seconds", 1.0)
        estimated_wait = service_time * (outstanding / capacity)
        candidates.append(((service_time + estimated_wait, kv, gpu), worker))
    if candidates:
        return min(candidates, key=lambda candidate: candidate[0])[1]
    if has_healthy:
        raise WorkersAtCapacityError(model)
    raise NoAvailableWorkerError(model)
