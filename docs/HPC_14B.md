# Qwen 14B comparison

Run in an existing GPU allocation. Keep the local SSH tunnel open. Ctrl+C stops
the foreground 0.5B server without exiting its allocation shell.

```bash
module load apptainer/1.4.5
apptainer exec "$HOME/vllm-runtime/containers/vllm.sif" python3 -c 'from huggingface_hub import snapshot_download; import os; snapshot_download("Qwen/Qwen2.5-14B-Instruct", local_dir=os.path.expanduser("~/vllm-runtime/models/qwen2.5-14b-instruct"))'
```

Copy the monitor from local PowerShell (repository root):

```powershell
scp experiments/monitor.py srk2211@insomnia.rcs.columbia.edu:~/vllm-runtime/monitor.py
```

Start the separate monitor command on the allocated compute node before starting
the model. Its output is in home storage, accessible through the login node.

```bash
apptainer exec --nv "$HOME/vllm-runtime/containers/vllm.sif" python3 -c 'import psutil; print(psutil.__version__)'
mkdir -p "$HOME/vllm-runtime/results"
apptainer exec --nv "$HOME/vllm-runtime/containers/vllm.sif" python3 "$HOME/vllm-runtime/monitor.py" --label qwen14b-worker --interval 0.5 --duration 7200 --disk-path /tmp --output "$HOME/vllm-runtime/results/qwen14b-worker-01.jsonl" > "$HOME/vllm-runtime/results/qwen14b-monitor-01.log" 2>&1 &
echo "Monitor PID: $!"
```

Use a fresh output filename on subsequent runs. Without --pid, CPU/disk readings
describe the visible host, not exclusively the inference process. GPU samples may
include other workloads visible inside the allocation; retain UUIDs and context.

```bash
apptainer exec --nv \
  --bind "$HOME/vllm-runtime/models/qwen2.5-14b-instruct:/models/qwen14b:ro" \
  "$HOME/vllm-runtime/containers/vllm.sif" \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /models/qwen14b --served-model-name tokenflow-model \
  --dtype bfloat16 --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 --gpu-memory-utilization 0.9 --enable-prefix-caching
```

After startup, verify model root through the local tunnel before benchmarking.
The model alias stays the same; its root must now be /models/qwen14b.

```powershell
curl.exe --fail --max-time 10 http://127.0.0.1:18101/v1/models
python experiments/telemetry_baseline.py --worker http://127.0.0.1:18101 --output results/qwen2.5-14B-single-gpu
```

The baseline performs 1,024 measured requests plus 32 warmups, using the unchanged
baseline.json, while collecting timestamped vLLM metrics. Physical GPU telemetry
requires the separate monitor above. After the sweep, stop only that monitor
using its saved PID, then retrieve the JSONL and log:

```powershell
scp srk2211@insomnia.rcs.columbia.edu:~/vllm-runtime/results/qwen14b-worker-01.jsonl results/qwen2.5-14B-single-gpu/
scp srk2211@insomnia.rcs.columbia.edu:~/vllm-runtime/results/qwen14b-monitor-01.log results/qwen2.5-14B-single-gpu/
python experiments/summarize.py results/qwen2.5-14B-single-gpu/serving --telemetry results/qwen2.5-14B-single-gpu/vllm-metrics.jsonl results/qwen2.5-14B-single-gpu/qwen14b-worker-01.jsonl
```

Record startup logs and the downloaded model revision. Check allocation time
remaining after downloading; do not start a sweep close to job expiry. If the
container lacks psutil or startup fails, preserve the error and fix that before
running the benchmark. The earlier 0.5B run lacks physical GPU telemetry, so that
portion cannot be compared directly without another monitored 0.5B run.
