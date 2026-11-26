"""HTTP behavior shared by the student-facing OpenAI endpoints."""

from collections.abc import Mapping

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.types import Message


_ERRORS = {
    400: ("invalid_request_error", "invalid_request"),
    401: ("authentication_error", "invalid_api_key"),
    413: ("invalid_request_error", "request_body_too_large"),
    422: ("invalid_request_error", "invalid_request"),
    429: ("rate_limit_error", "workers_at_capacity"),
    499: ("request_error", "client_disconnected"),
    502: ("server_error", "upstream_error"),
    503: ("server_error", "worker_unavailable"),
    504: ("server_error", "upstream_timeout"),
}


def error_response(
    status: int, message: str, *, code: str | None = None, param: str | None = None,
    headers: Mapping[str, str] | None = None, include_detail: bool = False,
) -> JSONResponse:
    kind, default_code = _ERRORS.get(status, ("request_error", "http_error"))
    body: dict[str, object] = {
        "error": {"message": message, "type": kind, "code": code or default_code, "param": param},
    }
    if include_detail:
        # Retain the field used by clients of the original FastAPI gateway.
        body["detail"] = message
    return JSONResponse(body, status_code=status, headers=headers)


def _limited_request(request: Request, limit: int) -> Request:
    """Enforce the cap before FastAPI buffers a declared or chunked upload."""
    try:
        declared = int(request.headers.get("content-length", "0"))
    except ValueError:
        declared = 0
    message = f"Inference request body exceeds the {limit}-byte limit."
    if declared > limit:
        raise HTTPException(413, message)

    received = 0

    async def receive() -> Message:
        nonlocal received
        event = await request.receive()
        if event["type"] == "http.request":
            received += len(event.get("body", b""))
            if received > limit:
                raise HTTPException(413, message)
        return event

    return Request(request.scope, receive)


class InferenceRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                if self.body_field is not None:
                    request = _limited_request(request, request.app.state.max_inference_body_bytes)
                return await handler(request)
            except RequestValidationError:
                # Pydantic errors may contain the client's complete input.
                return error_response(422, "The request body must be a valid JSON object.")
            except HTTPException as exc:
                message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
                return error_response(
                    exc.status_code, message, headers=exc.headers,
                    include_detail=isinstance(exc.detail, str) and exc.status_code != 413,
                )

        return handle
