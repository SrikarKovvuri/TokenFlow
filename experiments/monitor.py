"""Sample this host's resources to JSONL; run on the compute node for GPU data."""
import argparse
import csv
import io
import json
import math
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

UTC = timezone.utc  # Keep the standalone HPC monitor compatible with Python 3.9.

GPU_FIELDS = ["index", "uuid", "utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "power.draw", "temperature.gpu"]


def gpu_sample():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        devices = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != len(GPU_FIELDS):
                raise ValueError("Unexpected nvidia-smi columns")
            device = {"index": row[0].strip(), "uuid": row[1].strip()}
            for key, value in zip(GPU_FIELDS[2:], row[2:]):
                try:
                    device[key] = float(value)
                except ValueError:
                    device[key] = None
            devices.append(device)
        return {"gpus": devices}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"gpus": [], "gpu_error": f"{type(exc).__name__}: {exc}"}


class Sampler:
    def __init__(self, disk_path, pid=None):
        self.disk_path = disk_path
        self.root = psutil.Process(pid) if pid else None
        self.processes = {}
        psutil.cpu_percent(None)  # Prime interval-based CPU readings.
        self.previous = None
        self.previous_time = None

    def sample(self):
        started_wall = datetime.now(UTC)
        now = time.monotonic()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(self.disk_path)
        devices = psutil.disk_io_counters(perdisk=True) or {}
        network = psutil.net_io_counters()
        counters = {"network_sent_bytes": network.bytes_sent, "network_received_bytes": network.bytes_recv} if network else {}
        for name, value in devices.items():
            counters[f"disk:{name}:read_bytes"] = value.read_bytes
            counters[f"disk:{name}:write_bytes"] = value.write_bytes
        rates = {}
        if self.previous is not None:
            elapsed = now - self.previous_time
            rates = {key + "_per_second": (value - self.previous[key]) / elapsed
                     for key, value in counters.items() if key in self.previous and value >= self.previous[key]}
        self.previous, self.previous_time = counters, now
        row = {"timestamp": datetime.now(UTC).isoformat(), "hostname": socket.gethostname(),
               "platform": platform.platform(), "scope": "host", "cpu_percent": psutil.cpu_percent(None),
               "memory_used_bytes": memory.used, "memory_available_bytes": memory.available,
               "memory_percent": memory.percent, "disk_path": str(self.disk_path),
               "disk_used_bytes": disk.used, "disk_free_bytes": disk.free, "disk_percent": disk.percent,
               "counters": counters, "rates": rates}
        if self.root:
            try:
                processes = [self.root, *self.root.children(recursive=True)]
                stats = []
                active = {}
                for process in processes:
                    try:
                        key = (process.pid, process.create_time())
                        tracked = self.processes.get(key)
                        if tracked is None:
                            tracked = process
                            tracked.cpu_percent(None)
                            cpu = None
                        else:
                            cpu = tracked.cpu_percent(None)
                        active[key] = tracked
                        stats.append({"pid": tracked.pid, "cpu_percent": cpu,
                                      "rss_bytes": tracked.memory_info().rss})
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                self.processes = active
                row["process_tree"] = stats
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                row["process_error"] = str(exc)
        row.update(gpu_sample())
        finished_wall = datetime.now(UTC)
        row["sample_started_at"] = started_wall.isoformat()
        row["sample_finished_at"] = finished_wall.isoformat()
        row["timestamp"] = (started_wall + (finished_wall - started_wall) / 2).isoformat()
        return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True, help="For example worker or gateway")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--pid", type=int, help="Optional inference/gateway PID; sample it and descendants")
    parser.add_argument("--disk-path", default=".")
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.interval, args.duration)):
        parser.error("interval and duration must be finite and positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(args.disk_path, args.pid)
    deadline = time.monotonic() + args.duration
    try:
        with output.open("x", encoding="utf-8") as handle:
            first_sample = True
            while time.monotonic() < deadline:
                started = time.monotonic()
                row = sampler.sample()
                row["label"] = args.label
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if first_sample:
                    print(f"Monitoring {row['hostname']} at {row['timestamp']}; output={output}", flush=True)
                    print("Visible GPU UUIDs:", [g["uuid"] for g in row["gpus"]], flush=True)
                    if "gpu_error" in row:
                        print("GPU collection error:", row["gpu_error"], flush=True)
                    first_sample = False
                time.sleep(min(max(0, deadline - time.monotonic()),
                               max(0, args.interval - (time.monotonic() - started))))
    except KeyboardInterrupt:
        print(f"Stopped; samples saved to {output}")
