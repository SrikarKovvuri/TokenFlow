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

## Start small

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
- `suite-source.py`: snapshot of the runner used to generate the workload.
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
