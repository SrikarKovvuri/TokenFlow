import asyncio
import json
import logging
from collections.abc import Callable
from time import perf_counter
from uuid import UUID

import httpx

from tokenflow.worker_manager.models import HealthStatus, Worker
from tokenflow.worker_manager.registry import WorkerRegistry
from tokenflow.worker_manager.telemetry import read_telemetry


logger = logging.getLogger("uvicorn.error")


class WorkerHealthMonitor:
    """Poll registered inference servers and record their observed health."""

    def __init__(
        self,
        registry: WorkerRegistry,
        client: httpx.AsyncClient,
        *,
        interval_seconds: float,
        timeout_seconds: float,
        managed_worker: Callable[[UUID], bool] | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._interval_seconds = interval_seconds
        self._timeout_seconds = timeout_seconds
        self._managed_worker = managed_worker

    async def run(self) -> None:
        while True:
            await self.check_once()
            await asyncio.sleep(self._interval_seconds)

    async def check_once(self) -> None:
        workers = await self._registry.list()
        enabled_workers = [worker for worker in workers if worker.enabled]
        await asyncio.gather(*(self._poll(worker) for worker in enabled_workers))

    async def _poll(self, worker: Worker) -> None:
        try:
            await self._check_worker(worker)
        except Exception:
            # Keep polling the fleet if one endpoint supplies unexpected data.
            # Cancellation still propagates when the gateway shuts down.
            logger.exception("Unexpected health probe failure for worker %s", worker.id)

    async def _check_worker(self, worker: Worker) -> None:
        base = str(worker.base_url).rstrip("/")
        health_status = HealthStatus.UNHEALTHY

        started = perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._client.stream("GET", f"{base}/health", timeout=self._timeout_seconds) as response:
                    status_code = response.status_code
                if status_code != httpx.codes.OK:
                    logger.warning("Worker %s health check returned HTTP %d", worker.id, status_code)
                elif self._managed_worker and self._managed_worker(worker.id):
                    model = worker.metadata["model"]
                    if await self._serves_model(base, model):
                        health_status = HealthStatus.HEALTHY
                    else:
                        logger.warning("Worker %s did not advertise expected model %s", worker.id, model)
                else:
                    health_status = HealthStatus.HEALTHY
        except (httpx.HTTPError, TimeoutError) as exc:
            logger.warning(
                "Worker %s health check failed after %.3fs: %s: %r",
                worker.id, perf_counter() - started, type(exc).__name__, str(exc),
            )

        current = await self._registry.set_health_status(
            worker.id, health_status, expected_base_url=str(worker.base_url)
        )
        if current is None or health_status != HealthStatus.HEALTHY:
            return
        if current.metadata.get("backend") not in ("vllm", "sglang"):
            return

        engine_metrics = httpx.URL(f"{str(worker.base_url).rstrip('/')}/metrics")
        urls = [engine_metrics]
        gpu_port = current.metadata.get("gpu_metrics_port")
        if type(gpu_port) is int and 1 <= gpu_port <= 65535:
            urls.append(engine_metrics.copy_with(port=gpu_port, path="/metrics"))
        scrapes = await asyncio.gather(*(self._scrape(url) for url in urls))
        await self._registry.set_telemetry(
            worker.id, read_telemetry(*scrapes), expected_base_url=str(worker.base_url)
        )

    async def _read_bounded(self, url: str | httpx.URL, limit: int) -> bytes:
        async with self._client.stream("GET", url, timeout=self._timeout_seconds) as response:
            if response.status_code != 200:
                return b""
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > limit:
                    return b""
                body.extend(chunk)
            return bytes(body)

    async def _serves_model(self, base: str, model: str) -> bool:
        data = await self._read_bounded(f"{base}/v1/models", 1024 * 1024)
        try:
            catalog = json.loads(data)
        except (ValueError, RecursionError):
            return False
        models = catalog.get("data") if isinstance(catalog, dict) else None
        return isinstance(models, list) and any(isinstance(row, dict) and row.get("id") == model for row in models)

    async def _scrape(self, url: httpx.URL) -> str:
        """A missing or excessive metrics response cannot take a healthy worker out."""
        try:
            async with asyncio.timeout(self._timeout_seconds):
                body = await self._read_bounded(url, 4 * 1024 * 1024)
                return body.decode("utf-8", errors="replace")
        except (httpx.HTTPError, TimeoutError):
            pass
        return ""
