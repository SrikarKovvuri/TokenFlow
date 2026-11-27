"""Join complete benchmark cells with per-device and engine samples, using UTC windows."""
import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def gpu_samples(path):
    devices = defaultdict(list)
    if path.exists():
        with path.open() as handle:
            for row in csv.reader(handle, skipinitialspace=True):
                if len(row) != 11:
                    continue
                try:
                    timestamp = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                    sample = [timestamp, *map(float, row[3:7]), float(row[7])]
                except ValueError:
                    continue
                devices[row[2]].append(sample)
    return devices


def engine_samples(path):
    samples = []
    if path.exists():
        try:
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    sample = json.loads(line)
                    gauges = defaultdict(list)
                    for g in sample.get("gauges", []):
                        if g["value"] is not None:
                            gauges[g["name"].split(":")[-1]].append(g["value"])
                    samples.append(((sample["start_epoch"]+sample["end_epoch"])/2,
                                    {k:max(v) for k,v in gauges.items()}))
        except (EOFError, json.JSONDecodeError):
            # A live sync can end in the middle of the current gzip member.
            if (path.parent/"release.json").exists():
                raise
    return samples


def join(folder):
    worker = json.loads((folder/"worker.json").read_text())
    devices = gpu_samples(folder/"gpu.csv")
    engine = engine_samples(folder/"engine.jsonl.gz")
    for file in sorted(folder.glob("trial-*/c-*/summary.json")):
        cell = json.loads(file.read_text())
        start, end = cell["start_epoch"], cell["end_epoch"]
        gauges = [row for t,row in engine if start <= t < end and row]
        fields = {"num_requests_running":"running","num_requests_waiting":"waiting",
                  "kv_cache_usage_perc":"kv_fraction","gpu_cache_usage_perc":"kv_fraction"}
        stats = {}
        for field,label in fields.items():
            values = [v[field] for v in gauges if field in v]
            if values:
                stats[label+"_mean"] = statistics.mean(values)
                stats[label+"_peak"] = max(values)
        base = dict(controller=folder.name, gpu_job=worker["gpu_job"], tp=worker["tp"],
                    sequence_cap=worker["cap"], engine_samples=len(gauges),
                    output_tok_s_per_gpu=cell["output_tokens_per_second"]/worker["tp"], **cell, **stats)
        for uuid in worker["uuids"]:
            samples = devices[uuid]
            measured = [r for r in samples if start <= r[0] < end]
            idle = covered = 0.0
            for left,right in zip(samples,samples[1:]):
                dt = min(end,right[0])-max(start,left[0])
                if dt > 0 and 0 < right[0]-left[0] <= 2:
                    # Approximate the preceding interval using the busy-time sample.
                    covered += dt
                    idle += dt*(1-right[1]/100)
            row = dict(base,uuid=uuid,gpu_samples=len(measured),gpu_observed_s=covered,
                       gpu_coverage_fraction=covered/(end-start),gpu_idle_est_s=idle if covered else None)
            if measured:
                row.update(gpu_util_pct_mean=statistics.mean(r[1] for r in measured),
                           memory_util_pct_mean=statistics.mean(r[2] for r in measured),
                           vram_peak_mib=max(r[3] for r in measured),vram_total_mib=measured[0][4],
                           watts_mean=statistics.mean(r[5] for r in measured))
            yield row


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    args=parser.parse_args()
    rows=[row for f in sorted((args.campaign/"results").glob("*/worker.json")) for row in join(f.parent)]
    if not rows:
        raise SystemExit("No completed measured cells yet.")
    output=args.campaign/"cell-gpu-summary.csv"
    temporary=output.with_suffix(".tmp")
    with temporary.open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(f"{len(rows)} cell/device rows written to {output}")
