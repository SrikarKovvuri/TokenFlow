"""Exercise the public gateway over TCP with the official OpenAI client."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from tokenflow.config import Settings
from tokenflow.main import create_app

openai = pytest.importorskip("openai", reason="Install tokenflow[dev] for the SDK compatibility test")


@asynccontextmanager
async def loopback_server(app):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(app, log_config=None, access_log=False, ws="none")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("Local test server stopped during startup")
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)


def simulated_worker(first_chunk_received):
    worker = FastAPI()

    @worker.get("/health")
    async def health():
        return {"status": "ok"}

    @worker.post("/v1/chat/completions")
    async def complete(payload: dict):
        common = {"id": "chatcmpl-local", "model": payload["model"], "created": 1}
        word = payload["messages"][0]["content"]
        if not payload.get("stream"):
            message = {"role": "assistant", "content": f"Hello {word}"}
            return {**common, "object": "chat.completion", "choices": [
                {"index": 0, "message": message, "finish_reason": "stop"}
            ]}

        async def events():
            for content, finish in [("Hello ", None), (word, "stop")]:
                choice = {"index": 0, "delta": {"content": content}, "finish_reason": finish}
                chunk = {**common, "object": "chat.completion.chunk", "choices": [choice]}
                yield f"data: {json.dumps(chunk)}\n\n"
                # Only generate the second chunk after the SDK has parsed the first.
                await asyncio.wait_for(first_chunk_received.wait(), timeout=5)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return worker


def test_openai_models_chat_and_incremental_stream():
    async def scenario():
        received = asyncio.Event()
        admin_key = "local-operator-test-only"
        client_key = "local-inference-test-only"
        gateway = create_app(Settings(
            _env_file=None, platform_config=None, worker_secret=None, admin_key=admin_key,
            api_keys=[client_key], worker_health_interval_seconds=0.05,
        ))
        async with (
            loopback_server(simulated_worker(received)) as worker_url,
            loopback_server(gateway) as gateway_url,
            httpx.AsyncClient(trust_env=False) as admin,
            openai.AsyncOpenAI(base_url=f"{gateway_url}/v1", api_key=client_key, max_retries=0,
                               timeout=5, http_client=httpx.AsyncClient(trust_env=False)) as sdk,
        ):
            registration = await admin.post(
                f"{gateway_url}/v1/workers", headers={"Authorization": f"Bearer {admin_key}"},
                json={"name": "sdk-worker", "base_url": worker_url, "metadata": {"model": "local-qwen"}},
            )
            assert registration.status_code == 201
            async with asyncio.timeout(5):
                while not (models := await sdk.models.list()).data:
                    await asyncio.sleep(0.02)
            assert [model.id for model in models.data] == ["local-qwen"]
            prompt = {"model": models.data[0].id, "messages": [{"role": "user", "content": "world"}]}
            reply = await sdk.chat.completions.create(**prompt)
            assert reply.choices[0].message.content == "Hello world"
            assert reply.choices[0].finish_reason == "stop"

            async with await sdk.chat.completions.create(**prompt, stream=True) as stream:
                first = await anext(stream)
                assert first.choices[0].delta.content == "Hello "
                received.set()
                remainder = [chunk async for chunk in stream]
            assert len(remainder) == 1
            assert remainder[0].choices[0].delta.content == "world"
            assert remainder[0].choices[0].finish_reason == "stop"

    asyncio.run(scenario())
