import asyncio

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from test_openai_client import loopback_server
from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate
from tokenflow.worker_manager.registry import WorkerRegistry


def test_busy_streams_cannot_block_connections_to_an_idle_worker():
    """Use TCP to exercise the actual upstream connection pool behind admission."""
    async def scenario():
        busy = FastAPI()
        idle = FastAPI()
        streams_started, finish = asyncio.Event(), asyncio.Event()
        stream_count = 0

        @busy.post("/v1/chat/completions")
        async def stream():
            async def events():
                nonlocal stream_count
                stream_count += 1
                if stream_count == 100:
                    streams_started.set()
                yield b'data: {"choices":[]}\n\n'
                await finish.wait()
                yield b"data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        @idle.post("/v1/chat/completions")
        async def complete():
            return {"model": "idle", "choices": []}

        registry = WorkerRegistry()
        settings = Settings(_env_file=None, platform_config=None, admin_key=None, api_keys=[])
        gateway = create_app(settings, registry=registry, enable_health_monitor=False)
        async with (
            loopback_server(busy) as busy_url, loopback_server(idle) as idle_url,
            gateway.router.lifespan_context(gateway),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as client,
        ):
            for model, url, capacity in [("busy", busy_url, 100), ("idle", idle_url, 1)]:
                worker = await registry.create(WorkerCreate(
                    name=model, base_url=url, metadata={"model": model, "capacity": capacity},
                ))
                await registry.set_health_status(worker.id, HealthStatus.HEALTHY, expected_base_url=str(worker.base_url))
            # Shrink only the wait for an upstream connection so a regression fails quickly.
            gateway.state.worker_client.timeout.pool = 0.5
            path = "/v1/chat/completions"
            payload = {"model": "busy", "stream": True}
            streams = [asyncio.create_task(client.post(path, json=payload)) for _ in range(100)]
            try:
                await asyncio.wait_for(streams_started.wait(), timeout=15)
                assert (await client.post(path, json=payload)).status_code == 429
                response = await client.post(path, json={"model": "idle"})
                assert response.status_code == 200, response.text
                assert response.json()["model"] == "idle"
            finally:
                finish.set()
                responses = await asyncio.gather(*streams)
            assert all(r.status_code == 200 and r.content.endswith(b"data: [DONE]\n\n") for r in responses)
            assert all(w.active_requests == 0 for w in await registry.list())

    asyncio.run(scenario())
