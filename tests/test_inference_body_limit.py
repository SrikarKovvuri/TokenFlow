import asyncio
import json

import httpx
import pytest

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.worker_manager.models import HealthStatus, WorkerCreate
from tokenflow.worker_manager.registry import WorkerRegistry


@pytest.mark.parametrize("size_header", ["oversized", "missing", "understated", "boundary"])
def test_upload_limit_before_inference(size_header):
    async def scenario():
        body = json.dumps({
            "model": "qwen", "stream": True,
            "messages": [{"role": "user", "content": "x" * 65536}],
        }).encode()
        limit = len(body) if size_header == "boundary" else len(body) - 1
        registry = WorkerRegistry()
        registry.track_demand(["qwen"])
        worker = await registry.create(WorkerCreate(name="test", base_url="http://worker",
                                                   metadata={"model": "qwen"}))
        await registry.set_health_status(worker.id, HealthStatus.HEALTHY, expected_base_url=str(worker.base_url))
        calls = []

        def inference(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  content=b"data: hello\n\ndata: [DONE]\n\n")

        settings = Settings(_env_file=None, platform_config=None, admin_key=None,
                            api_keys=[], max_inference_body_bytes=limit)
        app = create_app(settings, registry=registry,
                         worker_transport=httpx.MockTransport(inference), enable_health_monitor=False)
        consumed = 0

        async def upload():
            nonlocal consumed
            for chunk in [body[:32768], body[32768:]]:
                consumed += 1
                yield chunk
            if size_header in ("missing", "understated"):
                pytest.fail("Gateway continued reading the upload after exceeding the limit")

        headers = {"Content-Type": "application/json", "X-Request-ID": "upload-test"}
        if size_header == "oversized":
            headers["Content-Length"] = str(len(body))
        elif size_header == "understated":
            headers["Content-Length"] = "10"
        async with (app.router.lifespan_context(app),
                    httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                      base_url="http://gateway") as client):
            response = await client.post("/v1/chat/completions", headers=headers, content=upload())

        assert response.headers["X-Request-ID"] == "upload-test"
        assert (await registry.get(worker.id)).active_requests == 0
        if size_header == "boundary":
            assert response.status_code == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            assert response.content == b"data: hello\n\ndata: [DONE]\n\n"
            assert calls == [json.loads(body)]
            assert registry.demand_since("qwen", 30) == 1
        else:
            assert response.status_code == 413
            assert response.json()["error"] == {
                "message": f"Inference request body exceeds the {limit}-byte limit.",
                "type": "invalid_request_error", "param": None, "code": "request_body_too_large",
            }
            assert not calls and registry.demand_since("qwen", 30) == 0
            assert consumed == (0 if size_header == "oversized" else 2)

    asyncio.run(scenario())
