"""One-command setup behavior and a full local fake-worker smoke run."""
import argparse
import asyncio
import importlib.util
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).parents[1]


def load():
    spec = importlib.util.spec_from_file_location("smoke_command", ROOT / "experiments" / "smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("models,requested,expected", [(["model"], None, "model"),
    (["a", "b"], "b", "b"), (["a", "b"], None, None), (["a"], "missing", None)])
def test_model_discovery(models, requested, expected):
    smoke = load()
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={"data": [{"id": model} for model in models]}))) as client:
            return await smoke.discover_model(client, "http://worker", requested)
    if expected is None:
        with pytest.raises(smoke.SetupError):
            asyncio.run(scenario())
    else:
        assert asyncio.run(scenario()) == expected


def test_existing_worker_is_reused_without_duplicate_registration():
    smoke = load()
    methods = []
    async def handler(request):
        methods.append(request.method)
        return httpx.Response(200, json=[{"id": "worker-id", "base_url": "http://worker/", "enabled": True}])
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await smoke.register_worker(client, "http://gateway", "http://worker")
    assert asyncio.run(scenario()) == "worker-id"
    assert methods == ["GET"]


def test_competing_worker_prevents_misleading_benchmark():
    smoke = load()
    async def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json=[{"id": "other", "base_url": "http://other/", "enabled": True, "name": "other"}])
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await smoke.register_worker(client, "http://gateway", "http://worker")
    with pytest.raises(smoke.SetupError, match="Other enabled workers"):
        asyncio.run(scenario())


def test_unrelated_service_is_not_reused():
    smoke = load()
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={"status": "ok", "service": "other"}))) as client:
            await smoke.gateway_running(client, "http://gateway")
    with pytest.raises(smoke.SetupError, match="not a healthy TokenFlow"):
        asyncio.run(scenario())


def test_temporary_gateway_is_stopped_when_setup_fails(tmp_path, monkeypatch):
    smoke = load()
    class Process:
        stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self, timeout):
            return 0
    process = Process()
    async def model(*args):
        return "model"
    async def absent(*args):
        return False
    async def failure(*args):
        raise smoke.SetupError("startup failed")
    monkeypatch.setattr(smoke, "discover_model", model)
    monkeypatch.setattr(smoke, "gateway_running", absent)
    monkeypatch.setattr(smoke, "start_gateway", lambda *args: process)
    monkeypatch.setattr(smoke, "wait_gateway", failure)
    args = argparse.Namespace(output=str(tmp_path / "run"), worker="http://worker", model=None,
                              gateway="http://127.0.0.1:8000", setup_timeout=1, telemetry=[])
    with pytest.raises(smoke.SetupError, match="startup failed"):
        asyncio.run(smoke.smoke(args))
    assert process.stopped


@pytest.mark.parametrize("baseline", [False, True])
def test_existing_gateway_is_left_running(tmp_path, monkeypatch, baseline):
    smoke = load()
    async def model(*args):
        return "model"
    async def present(*args):
        return True
    async def register(*args):
        return "worker"
    async def ready(*args):
        pass
    async def suite(args):
        assert args.smoke is (not baseline)
        assert Path(args.config).name == ("baseline.json" if baseline else "suite.json")
        Path(args.output).mkdir()
    monkeypatch.setattr(smoke, "discover_model", model)
    monkeypatch.setattr(smoke, "gateway_running", present)
    monkeypatch.setattr(smoke, "register_worker", register)
    monkeypatch.setattr(smoke, "wait_worker", ready)
    monkeypatch.setattr(smoke, "run_suite", suite)
    monkeypatch.setattr(smoke, "export", lambda *args: [])
    monkeypatch.setattr(smoke, "show_results", lambda *args: 0)
    def unexpected(*args):
        raise AssertionError("Must not start or stop an existing gateway")
    monkeypatch.setattr(smoke, "start_gateway", unexpected)
    args = argparse.Namespace(output=str(tmp_path / "run"), worker="http://worker", model=None,
                              gateway="http://127.0.0.1:8000", setup_timeout=1, telemetry=[])
    args.baseline = baseline
    assert asyncio.run(smoke.smoke(args)) == 0


def test_failed_measurements_return_nonzero(tmp_path, capsys):
    smoke = load()
    rows = [{"failures": 1, "warmup_failures": 0, "requests": 2, "successes": 1,
             "concurrency": 1, "stream": False, "latency_p95_seconds": .2,
             "first_content_p50_seconds": None, "successful_requests_per_second": 1}]
    assert smoke.show_results(rows, tmp_path) == 1
    assert "1/2 measured requests passed" in capsys.readouterr().out


def test_full_command_starts_gateway_and_benchmarks_fake_worker(tmp_path, monkeypatch, capsys):
    smoke = load()
    class Worker(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_body(self, body, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == "/v1/models":
                self.send_body(b'{"data":[{"id":"fake-model"}]}')
            elif self.path == "/health":
                self.send_body(b"{}")
            else:
                self.send_error(404)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("stream"):
                self.send_body(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                    b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                    b'data: [DONE]\n\n', "text/event-stream")
            else:
                self.send_body(b'{"choices":[{"message":{"content":"hello"}}],"usage":{"completion_tokens":1}}')
    server = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    monkeypatch.setenv("TOKENFLOW_WORKER_HEALTH_INTERVAL_SECONDS", ".05")
    config = json.loads((ROOT / "experiments" / "suite.json").read_text())
    config["cooldown_seconds"] = 0
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    actual_suite = smoke.run_suite
    async def fast_suite(args):
        args.config = str(path)
        await actual_suite(args)
    monkeypatch.setattr(smoke, "run_suite", fast_suite)
    output = tmp_path / "smoke"
    args = argparse.Namespace(output=str(output), worker=f"http://127.0.0.1:{server.server_port}",
        model=None, gateway=f"http://127.0.0.1:{port}", setup_timeout=15, telemetry=[])
    try:
        assert asyncio.run(smoke.smoke(args)) == 0
        assert (output / "summary.csv").exists()
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["model"] == "fake-model"
        assert len((output / "cells.jsonl").read_text().splitlines()) == 4
        text = capsys.readouterr().out
        assert "8/8 measured requests passed" in text
        assert "Stopped the temporary gateway" in text
        with socket.socket() as check:
            assert check.connect_ex(("127.0.0.1", port)) != 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
