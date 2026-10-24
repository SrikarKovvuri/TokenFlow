# bang: Qwen 0.5B concurrency results

This folder contains an unchanged copy of the completed qwen-telemetry-01 run.
The original remains under experiments/results/qwen-telemetry-01.

Collected on October 23, 2026, from 22:47:36 to 22:51:39 UTC (measured requests).
The load generator and temporary TokenFlow gateway ran locally on Windows.
Requests reached the HPC vLLM worker through the existing local SSH tunnel.

- 1,024 measured requests and 32 warmups passed across 16 test cells.
- Concurrency: 1, 4, 16, 32; JSON and SSE; two trials per setting.
- Context: 128 synthetic reference words plus the instruction; output cap: 128 tokens.
- 466 timestamped raw vLLM scrapes; zero scrape errors.

Start with serving/summary.csv. Individual requests and responses are in
serving/requests-*.jsonl, and cell time windows are in serving/cells.jsonl.
The raw /metrics responses are in vllm-metrics.jsonl. gateway.log records gateway
forwarding and HTTP responses. environment.json, token-preflight.json, and the
text source snapshots describe the run configuration and measurement code.

IMPORTANT: No HPC monitor file has been received. This folder does not contain
measured physical GPU utilization, VRAM, or compute-node CPU/disk telemetry.
The saved vLLM KV-cache gauges describe cache allocation use, not total VRAM use.
No physical-resource bottleneck has been established from these results.

SHA256SUMS.csv records the copied source-file hashes, verified against the originals.
