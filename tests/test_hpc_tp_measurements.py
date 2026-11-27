"""Check the live sweep's measurement boundaries without launching a worker."""
import asyncio
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hpc_tp_load import generate
from summarize_hpc_tp import join


@pytest.mark.parametrize("finish,done,passed", [("length", True, True), ("stop", True, False), ("length", False, False)])
def test_stream_requires_complete_fixed_length(finish, done, passed):
    async def exercise():
        deltas = [{"choices":[{"delta":{"content":"x"},"finish_reason":None}]},
                  {"choices":[{"delta":{},"finish_reason":finish}]},
                  {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":256}}]
        body = "".join("data: "+json.dumps(event)+"\n\n" for event in deltas)
        if done:
            body += "data: [DONE]\n\n"
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            row = await generate(client,"http://test","test",0)
            assert ("error" not in row) == passed
    asyncio.run(exercise())


def test_idle_estimate_excludes_unobserved_intervals(tmp_path):
    (tmp_path/"worker.json").write_text(json.dumps(dict(gpu_job="test",tp=2,cap=16,uuids=["A","B"])))
    folder = tmp_path/"trial-0"/"c-1"
    folder.mkdir(parents=True)
    (folder/"summary.json").write_text(json.dumps(dict(start_epoch=1,end_epoch=7,output_tokens_per_second=4)))
    with (tmp_path/"gpu.csv").open("w") as h:
        for sec,busy in [(0,0),(1,80),(2,50),(6,100),(7,100),(8,0)]:
            stamp=datetime.fromtimestamp(sec,timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")
            h.write(f"{stamp},0,A,{busy},2,100,200,40,1000,500,30\n")
    with gzip.open(tmp_path/"engine.jsonl.gz","wt") as h:
        h.write(json.dumps(dict(start_epoch=1.4,end_epoch=1.6,gauges=[
            dict(name="vllm:num_requests_running",value=2),dict(name="vllm:num_requests_running",value=2),
            dict(name="vllm:kv_cache_usage_perc",value=.1)]))+"\n")
    a,b=list(join(tmp_path))
    assert a["gpu_idle_est_s"] == .5
    assert a["gpu_observed_s"] == 2
    assert a["gpu_coverage_fraction"] == pytest.approx(2/6)
    assert a["running_peak"] == 2 and a["kv_fraction_peak"] == .1
    assert b["gpu_samples"] == 0 and b["gpu_idle_est_s"] is None
