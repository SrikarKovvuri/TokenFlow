import asyncio
import logging
from time import perf_counter

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
    ) -> None:
        self._registry = registry
        self._client = client
        self._interval_seconds = interval_seconds
        self._timeout_seconds = timeout_seconds

    async def run(self) -> None:
        while True:
            await self.check_once()
            await asyncio.sleep(self._interval_seconds)

    async def check_once(self) -> None:
        workers = await self._registry.list()
        enabled_workers = [worker for worker in workers if worker.enabled]
        await asyncio.gather(
            *(self._check_worker(worker) for worker in enabled_workers)
        )

    async def _check_worker(self, worker: Worker) -> None:
        health_url = f"{str(worker.base_url).rstrip('/')}/health"
        health_status = HealthStatus.UNHEALTHY

        started = perf_counter()
        try:
            response = await self._client.get(
                health_url,
                timeout=self._timeout_seconds,
            )
            if response.status_code == httpx.codes.OK:
                health_status = HealthStatus.HEALTHY
            else:
                logger.warning(
                    "Worker %s health check returned HTTP %d after %.3fs",
                    worker.id, response.status_code, perf_counter() - started,
                )
        except httpx.HTTPError as exc:
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

    async def _scrape(self, url: httpx.URL) -> str:
        """A missing or excessive metrics response cannot take a healthy worker out."""
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._client.stream("GET", url) as response:
                    if response.status_code == 200:
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > 4 * 1024 * 1024:
                                return ""
                            body.extend(chunk)
                        return body.decode("utf-8", errors="replace")
        except (httpx.HTTPError, TimeoutError):
            pass
        return ""
