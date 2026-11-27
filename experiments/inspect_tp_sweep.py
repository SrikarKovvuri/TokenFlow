"""Verify a TokenFlow TP sweep tarball locally. Exit zero for a complete full sweep."""

import argparse
import csv
import gzip
import json
import math
import re
import tarfile
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path


CONCURRENCIES = (1, 2, 4, 8, 16, 32)
EXPECTED = set(product(range(3), (1, 2), CONCURRENCIES))
MIN_REQUESTS, OUTPUT_TOKENS = 96, 256
CELL_PATH = re.compile(r"serving/trial-(\d+)-tp-(\d+)/c-(\d+)/(summary\.json|requests\.jsonl)")
SMOKE_PATH = re.compile(r"smoke/trial-0-tp-([12])/warmup\.json")
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
ERROR = re.compile(r"\b(?:\w*Error|\w*Exception):\s*\S.*|\bTimeoutError\b|\b(?:TIME LIMIT|OUT_OF_MEMORY)\b")


def successful(row):
    """The runner sets success only after [DONE]; it saves the other SSE events."""
    if not isinstance(row, dict) or row.get("success") is not True or row.get("status_code") != 200:
        return False
    usage = row.get("usage")
    if (not isinstance(usage, dict) or usage.get("completion_tokens") != OUTPUT_TOKENS
            or row.get("finish_reason") != "length"):
        return False
    content = finish = usage_event = False
    for event in row.get("events", []):
        data = event["data"]
        if "error" in data:
            return False
        usage_event |= data.get("usage") == usage
        for choice in data.get("choices", []):
            content |= bool(choice.get("delta", {}).get("content"))
            finish |= choice.get("finish_reason") == "length"
    return content and finish and usage_event


@dataclass
class Cell:
    good: int = 0
    failed: int = 0
    invalid: int = 0
    sequences: set[int] = field(default_factory=set)
    summary: dict = field(default_factory=dict)

    def read_requests(self, lines):
        for line in lines:
            try:
                row = json.loads(line)
                seq = row["sequence"]
                if type(seq) is not int or seq < 0 or seq in self.sequences:
                    raise ValueError("Invalid or duplicate sequence")
                self.sequences.add(seq)
                if row.get("success") is False:
                    self.failed += 1
                elif successful(row):
                    self.good += 1
                else:
                    self.invalid += 1
            except (ValueError, KeyError, TypeError, AttributeError):
                self.invalid += 1

    def verified(self, concurrency):
        elapsed = self.summary.get("elapsed_seconds")
        return (type(elapsed) in (float, int) and elapsed > 0 and math.isfinite(elapsed)
                and self.good >= MIN_REQUESTS and self.failed == self.invalid == 0
                and max(self.sequences, default=-1) == self.good - 1
                and self.summary.get("concurrency") == concurrency
                and self.summary.get("requests") == self.summary.get("successes") == self.good
                and self.summary.get("failures") == 0
                and self.summary.get("output_tokens") == self.good * OUTPUT_TOKENS)


def inspect_archive(path):
    cells, smoke = defaultdict(Cell), {}
    samples, allocated, problems = Counter(), set(), []
    seen, clues = set(), deque(maxlen=5)
    run_exit = None
    try:
        with gzip.open(path, "rb") as compressed, tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = re.sub(r"^(?:\./)?(?:sweep-\d+/)?", "", member.name)
                cell = CELL_PATH.fullmatch(name)
                warmup = SMOKE_PATH.fullmatch(name)
                if not (cell or warmup or name in {"exit-code.txt", "allocated-uuids.txt", "gpu-telemetry.csv"}
                        or name == "job.log" or name.endswith("/vllm.log")):
                    continue
                if name in seen:
                    problems.append(f"duplicate archive entry: {name}")
                    continue
                seen.add(name)
                with archive.extractfile(member) as data:
                    try:
                        if cell:
                            trial, tp, concurrency = map(int, cell.groups()[:3])
                            record = cells[trial, tp, concurrency]
                            if cell[4] == "requests.jsonl":
                                record.read_requests(data)
                            else:
                                summary = json.load(data)
                                if not isinstance(summary, dict):
                                    raise ValueError("expected a cell summary object")
                                record.summary = summary
                        elif warmup:
                            rows = json.load(data)
                            smoke[int(warmup[1])] = isinstance(rows, list) and len(rows) == 1 and successful(rows[0])
                        elif name == "exit-code.txt":
                            run_exit = int(data.read())
                        elif name == "allocated-uuids.txt":
                            allocated = {"GPU-" + line.decode().strip().removeprefix("GPU-") for line in data if line.strip()}
                        elif name == "gpu-telemetry.csv":
                            for row in csv.reader((line.decode() for line in data), skipinitialspace=True):
                                if (len(row) >= 4 and GPU_UUID.fullmatch(row[2]) and row[0].strip()
                                        and row[3].rstrip(" %").isdecimal() and int(row[3].rstrip(" %")) <= 100):
                                    samples[row[2]] += 1
                        else:
                            for raw in data:
                                match = ERROR.search(raw.decode(errors="replace").strip())
                                if match:
                                    clue = f"{name}: {match[0][:180]}"
                                    if clue not in clues:
                                        clues.append(clue)
                    except (ValueError, KeyError, TypeError, AttributeError, csv.Error) as exc:
                        problems.append(f"{name}: {exc}")
            # The tar reader stops at its end blocks. Finish gzip to check its CRC and trailer.
            while compressed.read(1024 * 1024):
                pass
    except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
        problems.append(f"archive read failed: {exc}")

    verified = {key for key, record in cells.items() if key in EXPECTED and record.verified(key[2])}
    print(f"Archive: {path}\nRecorded run exit: {run_exit if run_exit is not None else 'missing'}")
    for tp in (1, 2):
        state = "no completed warmup record" if tp not in smoke else "passed" if smoke[tp] else "failed"
        print(f"TP={tp} streaming smoke: {state}")
    print(f"Timed cells: {len(verified)}/{len(EXPECTED)} verified; need at least {MIN_REQUESTS} successful "
          f"requests per cell, each with {OUTPUT_TOKENS} output tokens and a matching summary.")
    print(f"Raw timed requests: {sum(c.good for c in cells.values())} verified, "
          f"{sum(c.failed for c in cells.values())} reported failures, "
          f"{sum(c.invalid for c in cells.values())} invalid or duplicate records.")
    for trial, tp in product(range(3), (1, 2)):
        missing = [str(c) for c in CONCURRENCIES if (trial, tp, c) not in cells]
        if missing:
            print(f"  Missing trial={trial} TP={tp}: concurrency {', '.join(missing)}")
    for key in sorted(cells.keys() - verified):
        trial, tp, concurrency = key
        cell = cells[key]
        status = "unexpected" if key not in EXPECTED else "incomplete/inconsistent"
        print(f"  {status} trial={trial} TP={tp} concurrency={concurrency}: {cell.good} verified, "
              f"{cell.failed} failed, {cell.invalid} invalid; summary {'present' if cell.summary else 'missing'}")
    print("Physical GPU telemetry (whole run):")
    for uuid in sorted(allocated | samples.keys()):
        print(f"  {uuid}: {samples[uuid]} samples" + ("" if uuid in allocated else " (not listed in allocation)"))
    telemetry_ok = len(allocated) == 2 and set(samples) == allocated and all(samples.values())
    if not telemetry_ok:
        print("  Could not confirm samples from exactly the two allocated GPUs.")
    ok = (run_exit == 0 and all(smoke.get(tp) for tp in (1, 2))
          and verified == EXPECTED == cells.keys() and telemetry_ok and not problems)
    if not ok:
        for clue in (*problems, *clues):
            print(f"  {clue}")
    print("Full sweep verified." if ok else "Full sweep incomplete or unverifiable.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="sweep-JOBID.tar.gz from the two-GPU job")
    raise SystemExit(0 if inspect_archive(parser.parse_args().archive) else 1)
