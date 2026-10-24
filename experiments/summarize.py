"""Export per-cell benchmark summaries and timestamp-matched resource aggregates."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def aggregate_samples(samples, start, end):
    rows = [r for r in samples if start <= datetime.fromisoformat(r["timestamp"]) <= end]
    values = {}
    def add(key, value):
        if isinstance(value, (int, float)):
            values.setdefault(key, []).append(value)
    for row in rows:
        if row.get("kind") == "vllm":
            prefix = "vllm:" + row["endpoint"]
            add(prefix + ":samples", 1)
            if "error" in row:
                add(prefix + ":scrape_errors", 1)
            add(prefix + ":scrape_seconds", row.get("scrape_seconds"))
            for metric in row.get("metrics", []):
                add(prefix + ":" + metric["name"] + metric["labels"], metric["value"])
            continue
        prefix = f"{row['label']}:{row['hostname']}"
        add(prefix + ":samples", 1)
        for key in ("cpu_percent", "memory_percent", "memory_used_bytes", "disk_percent", "disk_free_bytes"):
            add(prefix + ":" + key, row.get(key))
        for key, value in row.get("rates", {}).items():
            add(prefix + ":" + key, value)
        for gpu in row.get("gpus", []):
            for key in ("utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw", "temperature.gpu"):
                add(prefix + ":gpu:" + gpu["uuid"] + ":" + key, gpu.get(key))
        tree = row.get("process_tree", [])
        if tree:
            add(prefix + ":process_tree_rss_bytes", sum(p["rss_bytes"] for p in tree))
            if all(p["cpu_percent"] is not None for p in tree):
                add(prefix + ":process_tree_cpu_percent", sum(p["cpu_percent"] for p in tree))
    result = {}
    for key, series in values.items():
        if key.endswith((":samples", ":scrape_errors")):
            result[key] = len(series)
        else:
            result[key + ":observations"] = len(series)
            result[key + ":mean"] = sum(series) / len(series)
            result[key + ":max"] = max(series)
    return result


def export(run, telemetry):
    samples = []
    for path in telemetry:
        with Path(path).open(encoding="utf-8") as handle:
            samples.extend(json.loads(line) for line in handle if line.strip())
    rows = []
    with (run / "cells.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            cell = json.loads(line)
            cell["task"] = cell["task"]["id"]
            cell.update(aggregate_samples(samples, datetime.fromisoformat(cell["started_at"]),
                                          datetime.fromisoformat(cell["finished_at"])))
            rows.append(cell)
    if not rows:
        raise ValueError("No completed cells found")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = run / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} cells to {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--telemetry", nargs="*", default=[])
    args = parser.parse_args()
    export(args.run, args.telemetry)
