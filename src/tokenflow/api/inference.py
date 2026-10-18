from collections.abc import AsyncIterator
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from tokenflow.api.dependencies import get_registry, get_worker_client
from tokenflow.registry import NoAvailableWorkerError, WorkerRegistry


router = APIRouter(prefix="/v1", tags=["inference"])
Registry = Annotated[WorkerRegistry, Depends(get_registry)]
WorkerClient = Annotated[httpx.AsyncClient, Depends(get_worker_client)]


@router.post("/chat/completions")
async def create_chat_completion(
    payload: dict[str, Any],
    request: Request,
    registry: Registry,
    client: WorkerClient,
) -> Response:
    try:
        worker = await registry.get_available()
    except NoAvailableWorkerError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    worker_url = (
        f"{str(worker.base_url).rstrip('/')}/v1/chat/completions"
    )
    upstream_request = client.build_request(
        "POST",
        worker_url,
        json=payload,
        headers={"X-Request-ID": request.state.request_id},
    )

    try:
        upstream_response = await client.send(
            upstream_request,
            stream=True,
        )
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
    should_stream = payload.get("stream") is True

    if should_stream and upstream_response.is_success:
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
