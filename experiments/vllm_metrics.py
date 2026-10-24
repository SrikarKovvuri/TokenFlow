"""Timestamp raw vLLM /metrics scrapes alongside selected serving gauges."""
import asyncio
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

GAUGES = (
    "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_requests_swapped",
)
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def parse_gauges(body):
    selected = []
    for line in body.splitlines():
        match = SAMPLE.match(line)
        if not match or match[1] not in GAUGES:
            continue
        try:
            value = float(match[3])
            if not math.isfinite(value):
                value = None
        except ValueError:
            value = None
        selected.append({"name": match[1], "labels": match[2] or "", "value": value})
    return selected


class MetricsRecorder:
    def __init__(self, worker, output, interval=.5):
        self.worker = worker.rstrip("/")
        self.output = Path(output)
        self.interval = interval
        self.stop = asyncio.Event()
        self.task = None

    async def scrape(self):
        start_wall = datetime.now(UTC)
        start = perf_counter()
        row = {"kind": "vllm", "endpoint": self.worker,
               "scrape_started_at": start_wall.isoformat()}
        try:
            response = await self.client.get(self.worker + "/metrics")
            row["status_code"] = response.status_code
            row["server_date_header"] = response.headers.get("date")
            row["raw_prometheus"] = response.text
            response.raise_for_status()
            row["metrics"] = parse_gauges(response.text)
            available = {m["name"] for m in row["metrics"]}
            row["unavailable_gauges"] = [name for name in GAUGES if name not in available]
        except httpx.HTTPError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["metrics"] = []
        finished = datetime.now(UTC)
        row["scrape_finished_at"] = finished.isoformat()
        row["timestamp"] = (start_wall + (finished - start_wall) / 2).isoformat()
        row["scrape_seconds"] = perf_counter() - start
        self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()

    async def loop(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), self.interval)
            except TimeoutError:
                await self.scrape()

    async def __aenter__(self):
        self.handle = self.output.open("x", encoding="utf-8")
        self.client = httpx.AsyncClient(timeout=2, trust_env=False,
                                       limits=httpx.Limits(max_keepalive_connections=0))
        try:
            await self.scrape()
        except BaseException:
            await self.client.aclose()
            self.handle.close()
            raise
        self.task = asyncio.create_task(self.loop())
        return self

    async def __aexit__(self, *exc):
        self.stop.set()
        try:
            await self.task
            await self.scrape()
        finally:
            await self.client.aclose()
            self.handle.close()
