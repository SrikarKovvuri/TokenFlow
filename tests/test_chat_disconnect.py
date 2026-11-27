"""Drive the public ASGI endpoint with an independently disconnecting client."""
import asyncio
import json

import httpx
import pytest

from tokenflow.main import create_app
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate
from tokenflow.worker_manager.registry import WorkerRegistry


@pytest.mark.parametrize("failure", ["before-headers", "during-stream", "buffered-body", "upstream-read-error"])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_stream_reservation_lifetime_through_api(failure, spec_version):
    async def scenario():
        registry = WorkerRegistry()
        worker = await registry.create(WorkerCreate(
            name="one-slot", base_url="http://worker:8000", metadata={"model": "qwen", "capacity": 1},
        ))
        await registry.set_health_status(worker.id, HealthStatus.HEALTHY, expected_base_url=str(worker.base_url))
        upstream_started, first_chunk, disconnected, allow_upstream_error = (asyncio.Event() for _ in range(4))
        upstream_reading_body = asyncio.Event()
        upstream_closed = asyncio.Event()
        attempts = 0

        class WorkerStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                upstream_reading_body.set()
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                await allow_upstream_error.wait()
                raise httpx.ReadError("worker died after one chunk")

            async def aclose(self):
                upstream_closed.set()

        async def inference(request):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                return httpx.Response(200, json={"choices": [{"message": {"content": "slot available"}}]})
            upstream_started.set()
            if failure == "before-headers":
                try:
                    await asyncio.Event().wait()
                finally:
                    upstream_closed.set()
            return httpx.Response(200, stream=WorkerStream(), headers={"content-type": "text/event-stream"})

        app = create_app(registry=registry, worker_transport=httpx.MockTransport(inference), enable_health_monitor=False)
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
                 "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
                 "http_version": "1.1", "query_string": b"", "headers": [(b"content-type", b"application/json")]}
        payload = {"model": "qwen", "stream": failure != "buffered-body",
                   "messages": [{"role": "user", "content": "hi"}]}
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(event):
            if event["type"] == "http.response.body" and event.get("body", b"").startswith(b"data:"):
                first_chunk.set()

        async with app.router.lifespan_context(app), asyncio.timeout(3):
            task = asyncio.create_task(app(scope, receive, send))
            try:
                await upstream_started.wait()
                if failure == "buffered-body":
                    await upstream_reading_body.wait()
                elif failure != "before-headers":
                    await first_chunk.wait()
                assert (await registry.get(worker.id)).active_requests == 1
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                    full = await client.post("/v1/chat/completions", json=payload)
                assert full.status_code == 429
                if failure == "upstream-read-error":
                    allow_upstream_error.set()
                    with pytest.raises(httpx.ReadError):
                        await task
                else:
                    disconnected.set()
                    await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            assert upstream_closed.is_set() and attempts == 1
            assert (await registry.get(worker.id)).active_requests == 0
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                next_request = await client.post("/v1/chat/completions", json={"model": "qwen"})
            assert next_request.status_code == 200
            assert next_request.json()["choices"][0]["message"]["content"] == "slot available"
            assert (await registry.get(worker.id)).active_requests == 0

    asyncio.run(scenario())
