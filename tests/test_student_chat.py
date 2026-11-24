"""Run the student example with the official SDK and a local managed gateway."""

import asyncio
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tokenflow.config import Settings
from tokenflow.main import create_app

openai = pytest.importorskip("openai")
from test_managed_gateway import ADMIN, DEPLOY, SECRET, STUDENT, AllocationProvider, Events
from test_openai_client import loopback_server

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/student_chat.py"
stream_when_ready = runpy.run_path(str(EXAMPLE))["stream_when_ready"]


def test_readme_command_waits_for_cold_worker_then_streams(tmp_path):
    async def scenario():
        config = json.loads((DEPLOY / "insomnia.example.json").read_text())
        config.update(state_path="student.sqlite3", interval_seconds=.05)
        path = tmp_path / "platform.json"
        path.write_text(json.dumps(config))
        provider = AllocationProvider()
        attempts = []
        prompts = []

        async def engine(req):
            if req.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "qwen-14b"}]})
            if req.url.path == "/v1/chat/completions":
                prompts.append(json.loads(req.content)["messages"][0]["content"])
                return httpx.Response(200, stream=Events(), headers={"Content-Type": "text/event-stream"})
            return httpx.Response(200)

        app = create_app(
            Settings(_env_file=None, platform_config=path, admin_key=ADMIN, api_keys=[STUDENT],
                     worker_secret=SECRET, worker_health_interval_seconds=.05),
            providers={"slurm": provider}, worker_transport=httpx.MockTransport(engine),
        )

        @app.middleware("http")
        async def record_attempt(req, call_next):
            response = await call_next(req)
            if req.url.path == "/v1/chat/completions":
                attempts.append((response.status_code, response.headers.get("x-tokenflow-admission")))
            return response

        async with loopback_server(app) as url:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(EXAMPLE), "qwen-14b", "Hello", "--wait-seconds", "10",
                cwd=tmp_path, env={**os.environ, "OPENAI_BASE_URL": url + "/v1",
                                   "OPENAI_API_KEY": STUDENT, "NO_PROXY": "127.0.0.1"},
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        assert process.returncode == 0, stderr.decode(errors="replace")
        assert stdout.strip() == b"Hello" and b"Waiting" in stderr
        assert attempts[:-1] and all(attempt == (503, "waiting-for-worker") for attempt in attempts[:-1])
        assert attempts[-1] == (200, None)
        assert prompts == ["Hello"] and provider.launches == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("reply,expected", [
    ("worker-503", openai.APIStatusError), ("full", openai.APIStatusError),
    ("read-timeout", openai.APITimeoutError), ("after-stream", httpx.ReadError),
    ("wait-exceeds-deadline", TimeoutError),
])
def test_example_only_retries_safe_admission_within_deadline(reply, expected):
    attempts = []

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            raise httpx.ReadError("Stream disconnected")

    def respond(request):
        attempts.append(request)
        marker = {"X-TokenFlow-Admission": "waiting-for-worker", "Retry-After": "600"}
        if reply == "read-timeout":
            raise httpx.ReadTimeout("Uncertain result", request=request)
        if reply == "after-stream":
            return httpx.Response(200, stream=BrokenStream(), headers={"Content-Type": "text/event-stream"})
        if reply == "full":
            return httpx.Response(429, headers=marker)
        return httpx.Response(503, headers=marker if reply == "wait-exceeds-deadline" else {})

    with openai.OpenAI(base_url="http://test/v1", api_key=STUDENT,
                       http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(expected):
            with stream_when_ready(client, "qwen-14b", "Hello", wait_seconds=1) as stream:
                list(stream)
    assert len(attempts) == 1


def test_example_honors_retry_after_and_uses_backoff(monkeypatch):
    delays = []
    clock = SimpleNamespace(monotonic=lambda: sum(delays), time=lambda: sum(delays), sleep=delays.append)
    monkeypatch.setitem(stream_when_ready.__globals__, "time", clock)
    hints = iter(["4", "Thu, 01 Jan 1970 00:00:07 GMT", "malformed"])

    def respond(request):
        hint = next(hints, None)
        if hint is None:
            return httpx.Response(200, content=b"data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
        return httpx.Response(503, headers={"X-TokenFlow-Admission": "waiting-for-worker", "Retry-After": hint})

    with openai.OpenAI(base_url="http://test/v1", api_key=STUDENT,
                       http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with stream_when_ready(client, "qwen-14b", "Hello", wait_seconds=12) as stream:
            assert list(stream) == []
    assert delays == [4, 3, 4]
