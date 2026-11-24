from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from anyio import CancelScope
from fastapi import HTTPException, Response, status
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from tokenflow.worker_manager.models import Worker


class WorkerConnectionError(HTTPException):
    """Connection setup failed before the inference request could be sent."""


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
            await super().__call__(scope, receive, send)
        finally:
            await self._finish()


async def forward_chat_completion(
    *,
    worker: Worker,
    payload: dict[str, Any],
    request_id: str,
    client: httpx.AsyncClient,
    release: Callable[[], Awaitable[None]],
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

    worker_url = f"{str(worker.base_url).rstrip('/')}/v1/chat/completions"
    try:
        upstream_request = client.build_request(
            "POST",
            worker_url,
            json=payload,
            headers={"X-Request-ID": request_id},
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        await finish()
        code = 504 if isinstance(exc, httpx.ConnectTimeout) else 502
        raise WorkerConnectionError(
            status_code=code, detail="Could not connect to the inference worker"
        ) from exc
    except httpx.TimeoutException as exc:
        await finish()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The inference worker timed out",
        ) from exc
    except httpx.RequestError as exc:
        await finish()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The inference worker could not be reached",
        ) from exc
    except BaseException:
        await finish()
        raise

    response_headers = _response_headers(upstream_response)

    if payload.get("stream") is True and upstream_response.is_success:
        return _WorkerStreamingResponse(upstream_response, finish)

    try:
        content = await upstream_response.aread()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "The inference worker timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, "The inference worker response was interrupted") from exc
    finally:
        await finish()

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


def _response_headers(response: httpx.Response) -> dict[str, str]:
    allowed_headers = ("content-type", "cache-control", "retry-after")
    return {
        header: response.headers[header]
        for header in allowed_headers
        if header in response.headers
    }
