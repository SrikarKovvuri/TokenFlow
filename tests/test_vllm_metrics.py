"""Raw vLLM scrapes retain labels, unavailable fields, and measurement windows."""
import asyncio
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BODY = '''# HELP vllm:num_requests_running Number of running requests
vllm:num_requests_running{engine="0",model_name="qwen"} 4.0
vllm:num_requests_waiting{engine="0",model_name="qwen"} 2.0
vllm:gpu_cache_usage_perc{engine="0",model_name="qwen"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="qwen"} NaN
unrelated_metric 99
'''


def test_parser_preserves_series_and_missing_values():
    metrics = load("vllm_metrics").parse_gauges(BODY)
    assert len(metrics) == 4
    assert metrics[0]["labels"] == '{engine="0",model_name="qwen"}'
    assert metrics[2]["value"] == .25
    assert metrics[3]["value"] is None


def test_recorder_saves_full_raw_body_and_timestamps(tmp_path, monkeypatch):
    module = load("vllm_metrics")
    client_type = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=BODY)), **kwargs))
    output = tmp_path / "metrics.jsonl"
    async def scenario():
        async with module.MetricsRecorder("http://worker", output):
            pass
    asyncio.run(scenario())
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["raw_prometheus"] == BODY
        assert row["kind"] == "vllm"
        assert row["status_code"] == 200
        assert row["scrape_started_at"] <= row["timestamp"] <= row["scrape_finished_at"]
        assert "vllm:num_requests_swapped" in row["unavailable_gauges"]


def test_scrape_failure_is_saved_without_zero_metrics(tmp_path, monkeypatch):
    module = load("vllm_metrics")
    client_type = httpx.AsyncClient
    def offline(request):
        raise httpx.ConnectError("offline", request=request)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(offline), **kwargs))
    output = tmp_path / "metrics.jsonl"
    async def scenario():
        async with module.MetricsRecorder("http://worker", output):
            pass
    asyncio.run(scenario())
    row = json.loads(output.read_text().splitlines()[0])
    assert "ConnectError" in row["error"]
    assert row["metrics"] == []


def test_engine_metrics_join_keeps_engines_separate():
    parser = load("vllm_metrics")
    exporter = load("summarize")
    samples = [{"kind": "vllm", "endpoint": "http://worker", "timestamp": "2026-01-01T00:00:02+00:00",
                "metrics": parser.parse_gauges(BODY), "scrape_seconds": .1}]
    result = exporter.aggregate_samples(samples, datetime(2026, 1, 1, tzinfo=UTC),
                                         datetime(2026, 1, 1, 0, 0, 4, tzinfo=UTC))
    key = 'vllm:http://worker:vllm:num_requests_running{engine="0",model_name="qwen"}'
    assert result[key + ":mean"] == 4
    assert result[key + ":observations"] == 1
    assert not any('engine="1"' in key for key in result)


def test_telemetry_preflight_checks_every_prompt(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    module = load("telemetry_baseline")
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"count": 12, "max_model_len": 2048})
    config = {"tasks": [{"id": "task", "instruction": "Summarize"}],
              "prompt_words": [8, 16], "warmup_requests": 1, "requests_per_cell": 2,
              "max_tokens": [128], "seed": 42}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await module.preflight(client, "http://worker", "qwen", config)
    rows = asyncio.run(scenario())
    assert len(rows) == len(requests) == 6
    assert all(row["input_tokens"] + 128 <= row["max_model_len"] for row in rows)


def test_telemetry_preflight_rejects_context_overflow(monkeypatch):
    import pytest
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    module = load("telemetry_baseline")
    config = {"tasks": [{"id": "task", "instruction": "Summarize"}],
              "prompt_words": [8], "warmup_requests": 0, "requests_per_cell": 1,
              "max_tokens": [128], "seed": 42}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={"count": 2000, "max_model_len": 2048}))) as client:
            await module.preflight(client, "http://worker", "qwen", config)
    with pytest.raises(module.SetupError, match="context window"):
        asyncio.run(scenario())
