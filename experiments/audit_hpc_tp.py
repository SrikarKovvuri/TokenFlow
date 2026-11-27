"""Check each managed sweep against the raw streams, hardware samples and drain receipt."""
import argparse
import json
import math
from pathlib import Path

from summarize_hpc_tp import join


def audit(folder):
    worker = json.loads((folder / "worker.json").read_text())
    cap, tp = worker["cap"], worker["tp"]
    levels = [c for c in [1,2,4,8,16,32,64] if c <= cap]
    expected = {(t,c) for t in range(3) for c in levels}
    joined = list(join(folder))
    passed = set()
    issues = []
    attempts = successes = 0
    errors = {}
    for file in sorted(folder.glob("trial-*/c-*/summary.json")):
        cell = json.loads(file.read_text())
        key = (cell["trial"], cell["concurrency"])
        good = count = 0
        numbered = set()
        for line in (file.parent / "requests.jsonl").open():
            req = json.loads(line)
            count += 1
            numbered.add(req["index"])
            if "error" in req:
                code = str(req.get("http", "transport"))
                errors[code] = errors.get(code, 0)+1
            elif (req.get("http")==200 and req.get("done") and req.get("finish")=="length"
                  and req.get("usage",{}).get("completion_tokens")==256 and req.get("content_seconds")):
                good += 1
            else:
                issues.append(f"{key}: incomplete successful stream at index {req['index']}")
        devices = [r for r in joined if (r["trial"],r["concurrency"])==key]
        valid = (key in expected and good > 0 and len(numbered)==count and set(range(count))==numbered
                 and max(12,2*key[1])<=count==cell["requests"] and good==cell.get("successes",count)
                 and count-good==cell.get("failures",0) and good*256==cell["output_tokens"]
                 and cell["elapsed_seconds"]>=30 and math.isclose(cell["output_tokens_per_second"],
                     good*256/cell["elapsed_seconds"],rel_tol=1e-6)
                 and len(devices)==tp and len({r["uuid"] for r in devices})==tp
                 and all(r["gpu_samples"]>0 and r["engine_samples"]>0 and r.get("kv_fraction_peak") is not None
                         and r.get("running_peak") is not None and r.get("waiting_peak") is not None for r in devices))
        if valid:
            passed.add(key)
        else:
            issues.append(f"{key}: cell counts, measurements or telemetry incomplete")
        attempts += count
        successes += good
    release = folder/"release.json"
    drained = release.exists() and json.loads(release.read_text())["job"]==worker["gpu_job"]
    return dict(controller=folder.name,gpu_job=worker["gpu_job"],tp=tp,cap=cap,verified_cells=len(passed),
                expected_cells=len(expected),measured_requests=attempts,successful_requests=successes,
                reported_errors=errors,drain_verified=drained,
                complete=passed==expected and not issues and drained,
                missing_or_invalid=sorted(expected-passed),issues=issues)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    args=parser.parse_args()
    results=[audit(p.parent) for p in sorted((args.campaign/"results").glob("*/worker.json"))]
    print(json.dumps(results,indent=2))
