"""Exercise health-client connection reuse against a real HTTP/1.1 server."""
import asyncio
import threading

import httpx
import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate
from tokenflow.worker_manager.registry import WorkerRegistry


def test_health_probes_do_not_reuse_connections_closed_by_worker():
    class Worker(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        probes = 0
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.probes += 1
            if self.probes > 1:
                # Reproduce a peer accepting a request on an existing connection
                # then disconnecting without response headers.
                self.close_connection = True
                return
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    class RecordingRegistry(WorkerRegistry):
        def __init__(self):
            super().__init__()
            self.observed = []
        async def set_health_status(self, worker_id, health_status, **kwargs):
            self.observed.append(health_status)
            return await super().set_health_status(worker_id, health_status, **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    async def scenario():
        # Control: a pooled client reproduces the observed protocol error.
        async with httpx.AsyncClient(trust_env=False) as pooled:
            url = f"http://127.0.0.1:{server.server_port}/health"
            assert (await pooled.get(url)).status_code == 200
            with pytest.raises(httpx.RemoteProtocolError):
                await pooled.get(url)
        registry = RecordingRegistry()
        await registry.create(WorkerCreate(name="worker", base_url=f"http://127.0.0.1:{server.server_port}"))
        app = create_app(Settings(worker_health_interval_seconds=.02,
                                  worker_health_timeout_seconds=1), registry=registry)
        async with app.router.lifespan_context(app):
            async with asyncio.timeout(5):
                while len(registry.observed) < 4:
                    await asyncio.sleep(.01)
            assert registry.observed == [HealthStatus.HEALTHY] * len(registry.observed)
    try:
        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
