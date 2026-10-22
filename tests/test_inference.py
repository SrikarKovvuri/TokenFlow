from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tokenflow.main import create_app
from tokenflow.worker_manager.models import HealthStatus, Worker


class StubRegistry:
    def __init__(self, worker: Worker | None) -> None:
        self._worker = worker

    async def list(self) -> list[Worker]:
        if self._worker is None:
            return []
        return [self._worker]


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
    app = create_app(
        worker_transport=httpx.MockTransport(handler),
        enable_health_monitor=False,
    )
    app.state.worker_registry = StubRegistry(worker)
    return app


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


def test_no_available_worker_returns_service_unavailable() -> None:
    async def unexpected_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(f"Unexpected request to {request.url}")

    app = make_app(unexpected_handler, None)

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 503


def test_worker_http_error_is_passed_through() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Unknown model"}},
        )

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Unknown model"


def test_worker_connection_failure_returns_bad_gateway() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("worker is offline", request=request)

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 502


def test_worker_timeout_returns_gateway_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("worker timed out", request=request)

    app = make_app(handler, healthy_worker())

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 504
