from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException, Response, status
from fastapi.responses import StreamingResponse

from tokenflow.worker_manager.models import Worker


async def forward_chat_completion(
    *,
    worker: Worker,
    payload: dict[str, Any],
    request_id: str,
    client: httpx.AsyncClient,
) -> Response:
    """Relay one chat-completion request to the selected worker."""
    worker_url = f"{str(worker.base_url).rstrip('/')}/v1/chat/completions"
    upstream_request = client.build_request(
        "POST",
        worker_url,
        json=payload,
        headers={"X-Request-ID": request_id},
    )
    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The inference worker timed out",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The inference worker could not be reached",
        ) from exc

    response_headers = _response_headers(upstream_response)

    if payload.get("stream") is True and upstream_response.is_success:
        return StreamingResponse(
            _stream_upstream(upstream_response),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        content = await upstream_response.aread()
    finally:
        await upstream_response.aclose()

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def _stream_upstream(
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()


def _response_headers(response: httpx.Response) -> dict[str, str]:
    allowed_headers = ("content-type", "cache-control")
    return {
        header: response.headers[header]
        for header in allowed_headers
        if header in response.headers
    }
