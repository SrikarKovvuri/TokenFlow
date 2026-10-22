import asyncio

import httpx

from tokenflow.worker_manager.models import HealthStatus, Worker
from tokenflow.worker_manager.registry import WorkerRegistry


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

        try:
            response = await self._client.get(
                health_url,
                timeout=self._timeout_seconds,
            )
            if response.status_code == httpx.codes.OK:
                health_status = HealthStatus.HEALTHY
        except httpx.HTTPError:
            pass

        await self._registry.set_health_status(worker.id, health_status)
