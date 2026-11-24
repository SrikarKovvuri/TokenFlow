"""Read gauges for one inference worker and its allocated GPUs."""
import math
import re
from datetime import UTC, datetime

from tokenflow.worker_manager.models import WorkerTelemetry


# An endpoint represents one inference replica, possibly split across TP ranks.
# Take the busiest rank for each gauge; summing would count a TP request twice.
GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage",
    "sglang:num_running_reqs": "running",
    "sglang:num_queue_reqs": "waiting",
    "sglang:token_usage": "kv_cache_usage",
    "DCGM_FI_DEV_GPU_UTIL": "gpu_utilization",
}
SAMPLE = re.compile(r"^([\w:]+)(?:\{.*\})?\s+(\S+)(?:\s+\d+)?$")


def read_telemetry(*scrapes: str) -> WorkerTelemetry | None:
    gauges: dict[str, float | int] = {}
    for scrape in scrapes:
        for line in scrape.splitlines():
            match = SAMPLE.fullmatch(line)
            if not match or match[1] not in GAUGES:
                continue
            field = GAUGES[match[1]]
            try:
                value = float(match[2])
            except ValueError:
                continue
            if not math.isfinite(value) or value < 0:
                continue
            if field == "gpu_utilization":
                value /= 100
            if field in ("gpu_utilization", "kv_cache_usage") and value > 1:
                continue
            if field in ("running", "waiting") and not value.is_integer():
                continue
            gauges[field] = max(value, gauges.get(field, 0))
    if not gauges:
        return None
    return WorkerTelemetry(observed_at=datetime.now(UTC), **gauges)
