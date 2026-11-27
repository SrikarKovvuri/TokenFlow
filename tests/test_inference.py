import asyncio
from collections.abc import AsyncIterator, Callable

import httpx
import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.scheduler.proxy import forward_chat_completion
from starlette.requests import ClientDisconnect
from tokenflow.worker_manager.models import HealthStatus, Worker, WorkerCreate
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
    *, with_worker: bool = True,
) -> FastAPI:
    registry = WorkerRegistry()
    if with_worker:
        asyncio.run(register_healthy_worker(registry))
    return create_app(
        registry=registry,
        worker_transport=httpx.MockTransport(handler),
        enable_health_monitor=False,
    )


async def register_healthy_worker(registry: WorkerRegistry) -> Worker:
    worker = await registry.create(WorkerCreate(
        name="gpu-worker", base_url="http://worker:8101", metadata={"model": "test-model"},
    ))
    await registry.set_health_status(worker.id, HealthStatus.HEALTHY, expected_base_url=str(worker.base_url))
    return await registry.get(worker.id)


def test_non_streaming_request_is_forwarded() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://worker:8101/v1/chat/completions"
        assert request.headers["X-Request-ID"] == "request-123"
        assert request.content
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

    app = make_app(handler)

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

    app = make_app(handler)

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

    app = make_app(unexpected_handler, with_worker=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 503


@pytest.mark.parametrize("body", [b'{"model":"private-prompt-sentinel",', b'["private-prompt-sentinel"]'])
def test_public_request_validation_does_not_echo_the_body(client: TestClient, body: bytes) -> None:
    response = client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert b"private-prompt-sentinel" not in response.content
    assert set(response.json()) == {"error"}
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error" and error["message"]
    assert error["code"] == "invalid_request" and error["param"] is None


def test_operator_endpoints_keep_their_existing_error_format() -> None:
    key = "operator-test-key-123"
    app = create_app(Settings(_env_file=None, api_keys=["student-test-key-123"], admin_key=key),
                     enable_health_monitor=False)
    with TestClient(app) as client:
        denied = client.get("/v1/workers")
        headers = {"Authorization": f"Bearer {key}"}
        disabled = client.get("/v1/pools", headers=headers)
        invalid = client.post("/v1/workers", headers=headers, json={})
    assert denied.status_code == 401 and denied.headers["WWW-Authenticate"] == "Bearer"
    assert denied.json() == {"detail": "A valid bearer key is required"}
    assert disabled.status_code == 404 and disabled.json() == {"detail": "Managed pools are disabled"}
    assert invalid.status_code == 422 and set(invalid.json()) == {"detail"}
    assert isinstance(invalid.json()["detail"], list)


@pytest.mark.parametrize("stream", [False, True])
def test_plaintext_worker_error_is_relayed_byte_for_byte(stream: bool) -> None:
    body = b"Worker temporarily unavailable\r\n\xa3\r\n"
    headers = {"Content-Type": "text/plain; charset=iso-8859-1", "Cache-Control": "no-store",
               "Retry-After": "7", "X-TokenFlow-Admission": "waiting-for-worker"}
    app = make_app(lambda _: httpx.Response(503, content=body, headers=headers))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "test-model", "stream": stream})
    assert response.status_code == 503 and response.content == body
    for header in ("Content-Type", "Cache-Control", "Retry-After"):
        assert response.headers[header] == headers[header]
    assert "X-TokenFlow-Admission" not in response.headers
    assert asyncio.run(app.state.worker_registry.list())[0].active_requests == 0


@pytest.mark.parametrize("code", [400, 503])
def test_worker_http_error_is_passed_through_without_gateway_admission_marker(code) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            code,
            json={"error": {"message": "Unknown model"}},
            headers={"X-TokenFlow-Admission": "waiting-for-worker", "Retry-After": "10"},
        )

    app = make_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == code and "x-tokenflow-admission" not in response.headers
    assert response.headers["retry-after"] == "10"
    assert response.json()["error"]["message"] == "Unknown model"


def test_worker_connection_failure_returns_bad_gateway() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("worker is offline", request=request)

    app = make_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"
    assert asyncio.run(app.state.worker_registry.list())[0].active_requests == 0


def test_worker_timeout_returns_gateway_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("worker timed out", request=request)

    app = make_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"


@pytest.mark.parametrize("failure", ["headers", "body", "upstream", "close", "cancel"])
def test_stream_cleanup_releases_worker_on_failure(failure: str) -> None:
    """Drive the ASGI response: buffered TestClient calls miss send failures."""
    async def scenario() -> None:
        registry = WorkerRegistry()
        worker = await register_healthy_worker(registry)
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
