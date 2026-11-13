import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tokenflow.main import create_app
from tokenflow.scheduler.proxy import forward_chat_completion
from starlette.requests import ClientDisconnect
from tokenflow.worker_manager.models import HealthStatus, Worker
from tokenflow.worker_manager.registry import WorkerRegistry


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def make_app(
    handler: Callable[[httpx.Request], object],
    worker: Worker | None,
) -> FastAPI:
    registry = WorkerRegistry()
    if worker is not None:
        registry._workers[worker.id] = worker
    return create_app(
        registry=registry,
        worker_transport=httpx.MockTransport(handler),
        enable_health_monitor=False,
    )


def healthy_worker() -> Worker:
    now = datetime.now(UTC)
    return Worker(
        id=uuid4(),
        name="gpu-worker",
        base_url="http://worker:8101",
        enabled=True,
        health_status=HealthStatus.HEALTHY,
        metadata={"model": "test-model"},
        registered_at=now,
        updated_at=now,
    )


def test_non_streaming_request_is_forwarded() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://worker:8101/v1/chat/completions"
        assert request.headers["X-Request-ID"] == "request-123"
        assert request.content
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "request-123"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"


def test_streaming_request_is_relayed_and_closed() -> None:
    stream = TrackingStream(
        [
            b'data: {"token":"hello"}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=stream,
        )

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert stream.closed is True
    assert asyncio.run(app.state.worker_registry.list())[0].active_requests == 0


def test_no_available_worker_returns_service_unavailable() -> None:
    async def unexpected_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(f"Unexpected request to {request.url}")

    app = make_app(unexpected_handler, None)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 503


def test_worker_http_error_is_passed_through() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Unknown model"}},
        )

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Unknown model"


def test_worker_connection_failure_returns_bad_gateway() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("worker is offline", request=request)

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 502
    assert asyncio.run(app.state.worker_registry.list())[0].active_requests == 0


def test_worker_timeout_returns_gateway_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("worker timed out", request=request)

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 504


@pytest.mark.parametrize("failure", ["headers", "body", "upstream", "close", "cancel"])
def test_stream_cleanup_releases_worker_on_failure(failure: str) -> None:
    """Drive the ASGI response: buffered TestClient calls miss send failures."""
    async def scenario() -> None:
        registry = WorkerRegistry()
        worker = healthy_worker()
        registry._workers[worker.id] = worker
        releases = 0

        class Stream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b"data: hello\n\n"
                if failure in ("upstream", "close"):
                    raise httpx.ReadError("upstream interrupted")
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                # Real network cleanup may suspend under a cancellation scope.
                await anyio.sleep(0)
                self.closed = True
                if failure == "close":
                    raise OSError("upstream close failed")

        stream = Stream()

        async def release():
            nonlocal releases
            releases += 1
            await registry.release(worker.id)

        async def receive():
            await anyio.sleep_forever()

        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )) as client:
            response = await forward_chat_completion(
                worker=await registry.assign("test-model"),
                payload={"model": "test-model", "stream": True},
                request_id="cleanup-test", client=client, release=release,
            )
            assert (await registry.get(worker.id)).active_requests == 1
            with anyio.CancelScope() as scope:
                async def send(message):
                    assert (await registry.get(worker.id)).active_requests == 1
                    if failure == "headers" and message["type"] == "http.response.start":
                        raise OSError("client disconnected before body")
                    if message["type"] == "http.response.body":
                        if failure == "body":
                            raise OSError("client disconnected during body")
                        if failure == "cancel":
                            scope.cancel()
                            await anyio.sleep(0)

                if failure == "cancel":
                    await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
                else:
                    expected = (ClientDisconnect if failure in ("headers", "body")
                                else OSError if failure == "close" else httpx.ReadError)
                    with pytest.raises(expected):
                        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            assert stream.closed
            assert releases == 1
            assert (await registry.get(worker.id)).active_requests == 0

    asyncio.run(scenario())
