"""Bound the inference upload before FastAPI buffers and parses the JSON."""

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.types import Message


class _BodyTooLarge(HTTPException):
    # FastAPI preserves HTTP exceptions raised while it reads the request body.
    def __init__(self) -> None:
        super().__init__(413)


class InferenceRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def limited(request: Request) -> Response:
            limit = request.app.state.max_inference_body_bytes
            try:
                declared = int(request.headers.get("content-length", "0"))
            except ValueError:
                declared = 0
            received = 0

            async def receive() -> Message:
                nonlocal received
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > limit:
                        raise _BodyTooLarge()
                return message

            try:
                if declared > limit:
                    raise _BodyTooLarge()
                return await handler(Request(request.scope, receive))
            except _BodyTooLarge:
                return JSONResponse(status_code=413, content={"error": {
                    "message": f"Inference request body exceeds the {limit}-byte limit.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "request_body_too_large",
                }})

        return limited
