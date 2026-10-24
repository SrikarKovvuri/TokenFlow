"""Contract tests for experiment measurements, failures, and resource joins."""
import argparse
import asyncio
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(stream=True):
    return {"model": "test", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16, "stream": stream}


class Events(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data
        self.closed = False

    async def __aiter__(self):
        # Split inside JSON and across SSE boundaries to exercise incremental parsing.
        for index in range(0, len(self.data), 7):
            yield self.data[index:index + 7]

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("ending,expected_success", [(b"data: [DONE]\n\n", True), (b"", False)])
def test_sse_content_usage_and_truncation(ending, expected_success):
    suite = load("suite")
    data = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n' + ending)
    stream = Events(data)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream))) as client:
            return await suite.request_once(client, "http://test", payload(), 0, 1)

    row = asyncio.run(scenario())
    assert suite.success(row) is expected_success
    assert row["output"] == "hello"
    assert row["usage"]["completion_tokens"] == 2
    assert 0 <= row["time_to_first_content_seconds"] <= row["latency_seconds"]
    assert stream.closed


def test_request_deadline_is_recorded():
    suite = load("suite")

    async def handler(request):
        await asyncio.Event().wait()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await suite.request_once(client, "http://test", payload(False), 0, .01)

    row = asyncio.run(scenario())
    assert row["error"].startswith("TimeoutError")
    assert not suite.success(row)


def test_sweep_bounds_concurrency_and_exports_results(tmp_path, monkeypatch):
    suite = load("suite")
    config = json.loads((ROOT / "experiments" / "suite.json").read_text())
    config.update(tasks=config["tasks"][:1], prompt_words=[16], max_tokens=[16],
                  concurrency=[2], stream_modes=[False], trials=1, requests_per_cell=5,
                  warmup_requests=1, cooldown_seconds=0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "run"
    active = peak = count = 0

    async def handler(request):
        nonlocal active, peak, count
        active += 1
        peak = max(peak, active)
        count += 1
        try:
            await asyncio.sleep(.001)
            return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 3}})
        finally:
            active -= 1

    client_type = httpx.AsyncClient
    monkeypatch.setattr(suite.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handler), **kwargs))
    args = argparse.Namespace(config=str(path), smoke=False, dry_run=False,
                              output=str(output), model="test", target="http://test")
    asyncio.run(suite.run_suite(args))
    assert peak == 2
    assert count == 6
    cell = json.loads((output / "cells.jsonl").read_text())
    assert cell["requests"] == 5
    assert cell["successes"] == 5
    assert cell["warmup_failures"] == 0
    assert cell["usage_coverage_requests"] == 5
    assert cell["completion_tokens_per_second"] > 0
    assert len((output / "requests-0000.jsonl").read_text().splitlines()) == 5
    assert (output / "suite-source.txt").exists()
    assert not list(output.rglob("*.py")), "Result artifacts must not trigger Python reload watchers"
    exported = load("summarize").export(output, [])
    assert exported[0]["task"] == config["tasks"][0]["id"]
    assert (output / "summary.csv").exists()


def test_usage_gaps_do_not_invent_throughput():
    suite = load("suite")
    rows = [{"status_code": 200, "latency_seconds": 1, "usage": {"completion_tokens": 3}},
            {"status_code": 200, "latency_seconds": 2},
            {"status_code": 500, "latency_seconds": 99}]
    summary = suite.summarize(rows, 5)
    assert summary["completion_tokens_per_second"] is None
    assert summary["failures"] == 1
    assert summary["latency_p50_seconds"] == 1.5
    assert summary["successful_requests_per_second"] == .4


def test_resource_join_excludes_samples_outside_cell():
    exporter = load("summarize")
    rows = [{"timestamp": f"2026-01-01T00:00:0{second}+00:00", "label": "worker",
             "hostname": "gpu-node", "cpu_percent": cpu, "gpus": [
                 {"uuid": "gpu-1", "utilization.gpu": cpu, "power.draw": None}]} 
            for second, cpu in [(0, 100), (2, 20), (3, 40), (5, 100)]]
    metrics = exporter.aggregate_samples(rows, datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                                         datetime(2026, 1, 1, 0, 0, 4, tzinfo=UTC))
    assert metrics["worker:gpu-node:samples"] == 2
    assert metrics["worker:gpu-node:cpu_percent:mean"] == 30
    assert metrics["worker:gpu-node:gpu:gpu-1:utilization.gpu:max"] == 40
    assert not any("power.draw" in key for key in metrics)


def test_prompt_is_reproducible_and_has_requested_context_length():
    suite = load("suite")
    prompt = suite.prompt_for("Summarize", 128, "seed")
    assert prompt == suite.prompt_for("Summarize", 128, "seed")
    assert prompt != suite.prompt_for("Summarize", 128, "other")
    context = prompt.split("\n", 1)[1].split("\n\nTask:")[0]
    assert len(context.split()) == 128


@pytest.mark.parametrize("key,value", [("concurrency", [0]), ("stream_modes", ["true"]),
                                       ("requests_per_cell", -1), ("timeout_seconds", 0)])
def test_invalid_plan_fails_before_requests(key, value):
    suite = load("suite")
    config = json.loads((ROOT / "experiments" / "suite.json").read_text())
    config[key] = value
    with pytest.raises(ValueError):
        suite.validate(config)


def test_monitor_records_missing_gpu_tool(monkeypatch):
    pytest.importorskip("psutil")
    monitor = load("monitor")

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(monitor.subprocess, "run", missing)
    sample = monitor.gpu_sample()
    assert sample["gpus"] == []
    assert "FileNotFoundError" in sample["gpu_error"]

@pytest.mark.parametrize("body", [b"not json", b'data: []\n\ndata: [DONE]\n\n',
                                  b'data: {"choices":[{"delta":{"content":42}}]}\n\n'])
def test_bad_sse_is_a_recorded_failure(body):
    suite = load("suite")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Events(body)))) as client:
            return await suite.request_once(client, "http://test", payload(), 0, 1)

    row = asyncio.run(scenario())
    assert not suite.success(row)
    assert "error" in row


def test_dry_run_and_smoke_do_not_create_output(tmp_path, capsys):
    suite = load("suite")
    args = argparse.Namespace(config=str(ROOT / "experiments" / "suite.json"),
                              smoke=False, dry_run=True, output=str(tmp_path / "unused"))
    asyncio.run(suite.run_suite(args))
    assert "1152 cells, 73728 measured requests, 2304 warmup requests" in capsys.readouterr().out
    args.smoke = True
    asyncio.run(suite.run_suite(args))
    assert "4 cells, 8 measured requests, 4 warmup requests" in capsys.readouterr().out
    assert not (tmp_path / "unused").exists()


def test_gpu_sample_preserves_unavailable_fields(monkeypatch):
    pytest.importorskip("psutil")
    monitor = load("monitor")
    from types import SimpleNamespace
    monkeypatch.setattr(monitor.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="0, GPU-test, 85, 20, 4096, 8192, N/A, 55\n"))
    gpu = monitor.gpu_sample()["gpus"][0]
    assert gpu["utilization.gpu"] == 85
    assert gpu["memory.used"] == 4096
    assert gpu["power.draw"] is None


def test_monitor_derives_rates_after_first_sample(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    monitor = load("monitor")
    from types import SimpleNamespace
    times = iter([1, 3])
    devices = iter([{"disk0": SimpleNamespace(read_bytes=100, write_bytes=200)},
                    {"disk0": SimpleNamespace(read_bytes=300, write_bytes=600)}])
    monkeypatch.setattr(monitor.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(monitor.psutil, "disk_io_counters", lambda **kwargs: next(devices))
    monkeypatch.setattr(monitor, "gpu_sample", lambda: {"gpus": []})
    sampler = monitor.Sampler(str(tmp_path))
    first = sampler.sample()
    second = sampler.sample()
    assert first["rates"] == {}
    assert second["rates"]["disk:disk0:read_bytes_per_second"] == 100
    assert second["rates"]["disk:disk0:write_bytes_per_second"] == 200
