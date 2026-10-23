import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx


def test_experiment_records_responses_and_failures(tmp_path, monkeypatch):
    runner_path = Path(__file__).parents[1] / "experiments" / "run.py"
    spec = importlib.util.spec_from_file_location("experiment_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({"requests": [{"model": "ok"}, {"model": "bad"}]}))
    output = tmp_path / "results" / "run.json"

    async def handler(request):
        if json.loads(request.content)["model"] == "bad":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json={"choices": []}, headers={"X-Request-ID": "test"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(runner.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handler), **kwargs
    ))
    asyncio.run(runner.run(argparse.Namespace(
        workload=str(workload), output=str(output), gateway="http://gateway",
        timeout=1, concurrency=2, repeat=2,
    )))
    report = json.loads(output.read_text())
    assert report["summary"]["requests"] == 4
    assert report["summary"]["successes"] == 2
    assert report["summary"]["failures"] == 2
    assert report["results"][0]["request_id"] == "test"
    assert "ConnectError" in report["results"][1]["error"]
    assert all(r["latency_seconds"] >= 0 for r in report["results"])
