# Serving experiments

This directory holds client-side benchmarks and a separate host resource sampler.
The gateway and scheduler are unchanged. Start the worker, any required SSH tunnel,
and TokenFlow yourself; register the worker and wait for healthy status before a run.
No cluster provisioning, worker startup, registration, or automatic retries occur here.

## Install

Use your existing project Python environment:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -r experiments/requirements.txt
```

Only the resource monitor needs the additional `psutil` dependency. NVIDIA GPU
sampling uses `nvidia-smi` from the installed NVIDIA driver. Copy this directory to
the HPC compute environment to collect worker telemetry there. Run the monitor
inside the allocated compute node, rather than on its login node.

## One-command local smoke test

With your HPC worker and SSH tunnel already running, run this from the repo root:

```powershell
python experiments/smoke.py
```

The default worker URL is `http://127.0.0.1:1810` and the local gateway URL is
`http://127.0.0.1:8000`. For a different tunnel port:

```powershell
python experiments/smoke.py --worker http://127.0.0.1:YOUR_TUNNEL_PORT
```

This checks worker health, discovers the served model, reuses an existing healthy
TokenFlow gateway or starts a temporary local one, registers/reuses the worker,
waits for healthy status, runs the four smoke cells, exports `summary.csv`, and
prints pass/fail counts, latency, first-content timing, and successful requests/sec.
Results go into a new timestamped directory under `experiments/results`.
Setup errors or failed measured/warmup requests produce a nonzero exit code.
If several models are served, select one with `--model MODEL_ID`.

A gateway started by this command is stopped afterward, including on failure or
Ctrl+C. An existing gateway stays running. Its matching worker is reused and, if
necessary, enabled. The command refuses gateways with other enabled workers so
the test cannot silently route to a different worker. Use
`--gateway http://127.0.0.1:8001` for a separate temporary gateway in that case.
Gateway startup logs are saved beside the results directory as `*-gateway.log`.
The project dependencies must already be installed; this command does not install
packages or configure the HPC allocation, inference engine, or SSH tunnel.

The HPC monitor remains a separate command on the compute node:

```bash
python experiments/monitor.py --label worker --interval 1 --duration 1800 --disk-path /tmp --output experiments/results/worker-smoke-01.jsonl
```

After stopping the monitor and copying its JSONL file locally, add its samples to
the completed run with `summarize.py RUN_DIRECTORY --telemetry LOCAL_MONITOR_FILE`.
You can also pass already copied sample files to `smoke.py --telemetry FILE`;
only samples overlapping the new run's cell timestamps are matched.

## Manual smoke test

Set the model to the exact name served by your inference worker:

```powershell
python experiments/suite.py --model YOUR_SERVED_MODEL --smoke --dry-run
python experiments/suite.py --model YOUR_SERVED_MODEL --smoke --output experiments/results/smoke-01
python experiments/summarize.py experiments/results/smoke-01
```

The smoke test runs four cells: concurrency 1 and 2, each with JSON and SSE.
Each has one warmup and two measured requests. Inspect failures before the full run.
The target defaults to `http://localhost:8000`. Set `--target` to a gateway or a
worker's base URL to compare direct serving against gateway forwarding. Each run
requires a new output directory so earlier results cannot be overwritten.

## Full experiment design

`suite.json` defines a shuffled Cartesian sweep:

- Four tasks: summarization, reasoning, coding, and structured extraction.
- Context lengths: 64, 512, 2,048, and 8,192 **words**, plus the task instruction.
- Output caps: 64, 256, and 1,024 tokens.
- Concurrency: 1, 2, 4, 8, 16, and 32 simultaneous requests.
- Non-streaming JSON and streaming SSE.
- Two trials, 64 measured requests and two unmeasured warmups per cell.
- A two-second pause between cells and a 300-second whole-request deadline.

That is **1,152 cells, 73,728 measured requests, and 2,304 warmup requests**.
Runtime depends on model speed, output length, concurrency, and failures; this can
be a long GPU experiment. Copy and reduce the configuration for early iterations.
The seed determines cell order and generated reference records. The reference
records are synthetic component latency records; tasks test serving behavior,
not the quality of reasoning on a representative production dataset.

```powershell
python experiments/suite.py --config experiments/suite.json --model YOUR_SERVED_MODEL --dry-run
python experiments/suite.py --config experiments/suite.json --model YOUR_SERVED_MODEL --output experiments/results/full-01
```

Prompt lengths are deliberately labeled in words, not approximate token counts.
The server's returned `usage.prompt_tokens` is the authoritative token count saved
in individual results. Check your model's context window before using the largest
context/output combination. Output caps are limits: EOS or task completion may
produce fewer tokens. Requests exceeding the context window are recorded as failures.
Use extraction tasks as an example of naturally shorter outputs.

The client maintains a fixed number of consumers per cell (closed-loop load).
Each consumer issues its next request after the previous one finishes. It does
not simulate a specified arrival rate or an unlimited external queue. The final
few requests produce a concurrency tail. Keep requests per cell comfortably above
concurrency and increase repeats/trials for stable tail-latency estimates.

For each task and length, request indices get deterministic varied reference
records. The same corpus is reused across output caps, concurrency, mode, and trial
for comparison. Prefix caching can therefore affect results. Record your engine's
prefix-cache setting and compare cold versus warm serving separately. Warmup is
per cell, not a guarantee that every request prefix or model execution path is warm.
Cell order is shuffled with a fixed seed to reduce systematic order bias.

## Collect worker and gateway resources

Start monitoring before the suite. Keep it running until the suite ends. For a
remote HPC worker, run this **on the GPU compute node** in a separate terminal:

```bash
python experiments/monitor.py --label worker --interval 1 --duration 14400 --pid WORKER_PID --disk-path /tmp --output experiments/results/worker-01.jsonl
```

Replace `WORKER_PID` with the numeric vLLM parent process ID. If you do not know
it, omit `--pid` to collect host/GPU data only. Choose a disk path on the filesystem
you want to inspect, such as the model cache mount. `--duration` is a maximum in
seconds; Ctrl+C stops collection and preserves flushed samples.

Optionally run another monitor on the gateway host:

```powershell
python experiments/monitor.py --label gateway --interval 1 --duration 14400 --output experiments/results/gateway-01.jsonl
```

Each sample records:

- Host CPU utilization, RAM used/available, and memory utilization.
- Filesystem capacity used/free for `--disk-path`.
- Per-device disk read/write byte counters and bytes/sec; host network byte counters and bytes/sec.
- Each visible GPU's UUID, compute utilization, memory-controller utilization,
  VRAM used/total (MiB), power (watts), and temperature (Celsius).
- Optional selected process and descendants: PID, CPU utilization, and resident RAM.

These are measurements of the machine where the monitor runs. Host/GPU metrics
include other processes and jobs. GPU visibility may be restricted by your cluster;
verify UUIDs against your allocation. Process CPU uses psutil's convention where
100% is one fully occupied logical CPU and a multi-threaded process may exceed
100%. Summed process RSS can double-count shared memory. Newly observed processes
have a null CPU sample until the next observation. Host CPU's initial call is
primed before sampling. Disk/network rates are null/absent until two observations
exist. Device counters describe the whole device, not just the selected filesystem;
do not sum partitions, logical devices, and their backing devices indiscriminately.
Disk capacity percent is not disk I/O utilization or I/O wait.

Missing `nvidia-smi`, unsupported GPU fields (including MIG restrictions), and driver
errors are recorded as `gpu_error` or null fields instead of being invented as zero.
CPU-only monitoring still works. NVIDIA queries have a five-second timeout;
actual sampling cadence can be slower than requested. No privileged access is
requested and no NVIDIA settings are changed.

## Extract and compare results

Copy the worker telemetry file back to the machine with the benchmark results.
Synchronize clocks on gateway, load generator, and compute node; timestamps use UTC.
The exporter joins telemetry to measured cell windows using those wall-clock times:

```powershell
python experiments/summarize.py experiments/results/full-01 --telemetry experiments/results/worker-01.jsonl experiments/results/gateway-01.jsonl
```

Output files:

- `manifest.json`: resolved configuration, target/model, seeded cell order, and load model.
- `suite-source.txt`: snapshot of the runner used to generate the workload. This is a text file so Python reload watchers do not restart the gateway and clear its registry.
- `warmup-NNNN.json`: warmup responses, kept outside measured summaries.
- `requests-NNNN.jsonl`: one flushed result per measured request, including timestamps,
  prompt hash and character count, request ID, status/error, response or streamed
  content, server usage when available, latency, and streaming first-content latency.
- `cells.jsonl`: one flushed completed-cell summary, including dimensions, UTC
  measurement window, warmup failures, success/failure counts, latency p50/p95/p99,
  streaming first-content p50/p95, successful requests/sec, and completion tokens/sec.
- `summary.csv`: cell summaries plus host/GPU mean and maximum resource values and
  matching sample counts when telemetry was provided. Run the exporter again to refresh it.

First-content latency measures the first nonempty SSE `delta.content` seen by the
client, not an engine-internal first-token timestamp. Role-only events do not count.
Token throughput uses server usage, never counts chunks or words as tokens. It is
null unless every successful request reports completion tokens. Stream requests
ask for `stream_options.include_usage`; check that your serving engine supports it.
Malformed/truncated streams and transport failures count as failed requests.
Latency percentiles include successful requests only; always inspect failures too.
Cell throughput includes client work and request-file writes, so monitor the load
generator for saturation before attributing a throughput ceiling to the GPU.

Resource aggregates are sample means/maxima, not time-weighted measurements. Short
cells may have no samples. Empty CSV fields mean unavailable, not zero. Warmups
and cooldowns are excluded from the cell joins. Partial interrupted request files
remain useful, but the exporter includes completed cells only. This suite does
not resume a run automatically or evaluate response correctness.

Record the GPU model/count, allocation, inference engine/version, model revision,
quantization, tensor parallelism, context window, cache settings, gateway version,
and tunnel/network topology in your experiment notes. Keep these constant when
comparing one dimension. Save a separate run for direct-worker measurements.
Generated results are Git-ignored and can contain prompts, responses, machine
names, and runtime addresses.

The smaller original runner remains available for custom request lists:

```powershell
python experiments/run.py --workload experiments/workload.example.json --repeat 10 --concurrency 4 --output experiments/results/custom-01.json
```

Resource APIs follow the [psutil documentation](https://psutil.readthedocs.io/stable/)
and [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Reusable model-comparison baseline

With a running worker and tunnel, use the same setup command with `--baseline`:

```powershell
python experiments/smoke.py --baseline --worker http://127.0.0.1:YOUR_TUNNEL_PORT
```

`baseline.json` pins one summarization task, 128 context words, a 128-token output
cap, concurrency 1/4/16/32, JSON and SSE, and two trials. Each cell has 64 measured
requests and two warmups: 16 cells, 1,024 measured requests, and 32 warmups total.
The printed summary and CSV use the same metrics as the smoke test. Results use
new timestamped baseline directories by default. Reuse this configuration after
switching models and record the actual server-reported token counts and output
lengths; the same text does not imply identical tokenization across models.

Before comparing models, predict how throughput and latency will change with
concurrency. Inspect failures first, then compare cells with the same prompt,
output cap, mode, and concurrency. Warmup and prefix-cache behavior can affect
both trials. This is a short engineering baseline, not a claim of peak GPU
throughput or a model-quality evaluation. Record hardware/engine settings and
collect compute-node telemetry separately before attributing changes to GPU use.

Health probes use a dedicated HTTP client with idle connection reuse disabled.
This avoids false unhealthy states from reused connections that a worker or SSH
forwarding path has closed. Failed probes still mark workers unhealthy; logs now
include the HTTP status or transport exception and elapsed time. No inference
retry or timeout increase is used to hide errors. The source snapshot uses `.txt`
so writing experiment results does not trigger Python reload watchers. Prefer a
gateway without `--reload` when benchmarking.

## Concurrency sweep with simultaneous telemetry

Run the unchanged `baseline.json` again with live vLLM scrapes using:

```powershell
python experiments/telemetry_baseline.py --worker http://127.0.0.1:YOUR_TUNNEL_PORT --output experiments/results/qwen-telemetry-01
```

Keep the HPC monitor as a separate process. Before the benchmark begins, start it
on the allocated GPU compute node, using the current `experiments/monitor.py`:

```bash
python experiments/monitor.py --label qwen-worker --interval 0.5 --duration 3600 --disk-path /tmp --output /tmp/tokenflow-qwen-telemetry-01.jsonl
```

If necessary, copy `monitor.py` and `requirements.txt` to the cluster first and
install the latter in your monitoring environment. Use `~/monitor.py` instead of
`experiments/monitor.py` if that is where you copied the script. Do not run the
resource monitor on the login node. Verify the visible GPU UUIDs belong to your
allocation. Clocks on the load generator and compute node must be synchronized.

Start the monitor **10 seconds before confirming readiness for the benchmark**.
Keep it running through setup, all 16 measured cells, and the final idle period.
The benchmark collects vLLM metrics for 10 seconds before and after the sweep.
Stop the HPC monitor with Ctrl+C **after the benchmark prints its final results**.
Share or copy back this exact file:

```text
/tmp/tokenflow-qwen-telemetry-01.jsonl
```

The local run directory contains `vllm-metrics.jsonl` (full raw Prometheus responses,
selected gauges, labels, UTC scrape start/end/midpoint, HTTP status or scrape error),
`environment.json` (model and engine identity/configuration), and
`token-preflight.json` (actual worker token counts for the workload). The
`serving/` subdirectory contains raw request JSONL files, warmup results,
`manifest.json`, timestamped cell windows in `cells.jsonl`, and `summary.csv`.

After copying the HPC file locally, join both sources to the measured cell windows:

```powershell
python experiments/summarize.py experiments/results/qwen-telemetry-01/serving --telemetry experiments/results/qwen-telemetry-01/vllm-metrics.jsonl experiments/results/tokenflow-qwen-telemetry-01.jsonl
```

Each CSV row keeps concurrency, JSON/SSE mode, trial, throughput, and latency next
to per-series mean/max resource and engine gauges. Observation counts expose
missing coverage. `vllm:gpu_cache_usage_perc` and `vllm:kv_cache_usage_perc` are
retained separately when both are exported; they are fractions (0 to 1) of the KV
cache allocation, not percentages of total VRAM. They must not be summed. Running
and waiting request counts retain engine/model labels; gauges are not combined
across engines automatically. GPU VRAM used/total is in MiB, GPU compute utilization
is percent, and host resource measurements describe the compute node as a whole.

Unavailable gauges and scrape failures are saved explicitly; missing data is not
zero. A 0.5-second scrape interval does not guarantee the engine refreshes its
underlying gauges at that rate. Short cells can miss activity peaks or contain
repeated/stale gauge values. Aggregates use sample means/maxima, not time-weighted
means. Inspect raw UTC samples, counts, and refresh cadence before attributing a
bottleneck; throughput alone is insufficient evidence of GPU saturation or KV
cache pressure. Gauge-only scrapes do not establish causation.
