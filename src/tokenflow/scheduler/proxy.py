import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from anyio import CancelScope
from fastapi import HTTPException, Response, status
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from tokenflow.worker_manager.models import Worker


class WorkerConnectionError(HTTPException):
    """Connection setup failed before the inference request could be sent."""


class _PeerDisconnected(Exception):
    pass


@asynccontextmanager
async def _while_connected(receive: Receive | None) -> AsyncIterator[None]:
    """After FastAPI consumes the JSON body, the next ASGI event is disconnect."""
    if receive is None:
        yield
        return

    with CancelScope() as scope:
        async def watch() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass
            scope.cancel()

        watcher = asyncio.create_task(watch())
        try:
            yield
        finally:
            watcher.cancel()
            with CancelScope(shield=True):
                await asyncio.wait((watcher,))
                if not watcher.cancelled():
                    watcher.result()
    if scope.cancel_called:
        raise _PeerDisconnected


class _WorkerStreamingResponse(StreamingResponse):
    """Own upstream cleanup even if sending to the client fails."""

    def __init__(
        self, upstream: httpx.Response, finish: Callable[[], Awaitable[None]]
    ) -> None:
        super().__init__(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
        )
        self._finish = finish

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            # On ASGI 2.4+ Starlette observes disconnects on send. An engine
            # can be silent between chunks, so also listen during that wait.
            spec = scope.get("asgi", {}).get("spec_version", "2.0")
            modern_asgi = tuple(map(int, spec.split("."))) >= (2, 4)
            async with _while_connected(receive if modern_asgi else None):
                await super().__call__(scope, receive, send)
        except _PeerDisconnected:
            pass
        finally:
            await self._finish()


async def forward_chat_completion(
    *,
    worker: Worker,
    payload: dict[str, Any],
    request_id: str,
    client: httpx.AsyncClient,
    release: Callable[[], Awaitable[None]],
    receive: Receive | None = None,
) -> Response:
    """Relay one chat-completion request to the selected worker.

    ``release`` runs once the upstream response is finished, has failed, or
    the stream is closed. A streaming response stays reserved until then.
    """
    released = False
    upstream_response: httpx.Response | None = None

    async def finish() -> None:
        nonlocal released
        if released:
            return
        released = True
        # A disconnect can cancel the response task. Finish cleanup anyway,
        # and release the reservation even if closing the upstream fails.
        with CancelScope(shield=True):
            try:
                if upstream_response is not None:
                    await upstream_response.aclose()
            finally:
                await release()

    stream_owns_reservation = False
    try:
        async with _while_connected(receive):
            upstream_request = client.build_request(
                "POST", f"{str(worker.base_url).rstrip('/')}/v1/chat/completions",
                json=payload, headers={"X-Request-ID": request_id},
            )
            try:
                upstream_response = await client.send(upstream_request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                code = 504 if isinstance(exc, httpx.ConnectTimeout) else 502
                raise WorkerConnectionError(code, "Could not connect to the inference worker") from exc

            if payload.get("stream") is True and upstream_response.is_success:
                response = _WorkerStreamingResponse(upstream_response, finish)
            else:
                response = Response(
                    content=await upstream_response.aread(),
                    status_code=upstream_response.status_code,
                    headers=_response_headers(upstream_response),
                )
        # Ownership passes only after the disconnect watcher has finished.
        stream_owns_reservation = isinstance(response, _WorkerStreamingResponse)
        return response
    except _PeerDisconnected as exc:
        # The peer has gone away. Supply a status for access logs and middleware.
        raise HTTPException(499, "Client disconnected") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The inference worker timed out",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The inference worker request was interrupted",
        ) from exc
    finally:
        if not stream_owns_reservation:
            await finish()


def _response_headers(response: httpx.Response) -> dict[str, str]:
    allowed_headers = ("content-type", "cache-control", "retry-after")
    return {
        header: response.headers[header]
        for header in allowed_headers
        if header in response.headers
    }
