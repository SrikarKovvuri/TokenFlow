#!/usr/bin/env bash
# Inside the vLLM container. All traffic stays on loopback.
set -euo pipefail
mode=$1
out=/run-output
worker_port=${TOKENFLOW_HPC_WORKER_PORT:-18000}
gateway_port=${TOKENFLOW_HPC_GATEWAY_PORT:-18001}
worker="http://127.0.0.1:$worker_port"
gateway="http://127.0.0.1:$gateway_port"
server=
monitor=
venv_dir=
cleanup() {
    rc=$?
    trap - EXIT
    for pid in "$monitor" "$server"; do
        if [[ -n $pid ]]; then kill "$pid" 2>/dev/null || true; fi
    done
    for pid in "$monitor" "$server"; do
        if [[ -n $pid ]]; then wait "$pid" 2>/dev/null || true; fi
    done
    if [[ -n $venv_dir ]]; then rm -rf -- "$venv_dir"; fi
    printf '%s\n' "$rc" > "$out/exit-code.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 - "$worker_port" "$gateway_port" <<'PY'
import socket, sys
with socket.socket() as worker, socket.socket() as gateway:
    for sock, port in zip((worker, gateway), sys.argv[1:]):
        sock.bind(("127.0.0.1", int(port)))
PY
{
    date -u
    hostname
    sha256sum pyproject.toml scripts/hpc-*.sh experiments/*.py experiments/*.json
    find src/tokenflow -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    nvidia-smi -L
} > "$out/run-info.txt"
if ! python3 -c 'import fastapi, httpx, pydantic_settings, psutil, uvicorn'; then
    # Keep gateway dependencies separate from the vLLM interpreter.
    venv_dir=$(mktemp -d /tmp/tokenflow-gateway.XXXXXXXX)
    python3 -m venv --system-site-packages "$venv_dir"
    "$venv_dir/bin/python" -c 'import subprocess, sys, tomllib; deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]; subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *deps, "psutil>=6,<8"])' \
        > "$out/dependencies.log" 2>&1
    runner="$venv_dir/bin/python"
else
    runner=python3
fi
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/qwen14b --served-model-name tokenflow-model \
    --host 127.0.0.1 --port "$worker_port" --dtype bfloat16 \
    --max-model-len 2048 --gpu-memory-utilization 0.9 --enable-prefix-caching \
    > "$out/vllm.log" 2>&1 &
server=$!
"$runner" experiments/monitor.py --label worker --interval 0.5 --duration 86400 \
    --pid "$server" --disk-path /tmp --output "$out/hardware.jsonl" \
    > "$out/monitor.log" 2>&1 &
monitor=$!
ready=false
for ((i=0; i<180; i++)); do
    kill -0 "$server" "$monitor" || { tail -n 30 "$out/vllm.log" "$out/monitor.log"; exit 1; }
    if curl -fsS --max-time 2 "$worker/health" >/dev/null 2>&1 && [[ -s $out/hardware.jsonl ]]; then
        ready=true
        break
    fi
    sleep 2
done
if [[ $ready != true ]]; then echo "Worker or monitor did not become ready." >&2; exit 1; fi
export PYTHONDONTWRITEBYTECODE=1
if [[ $mode == smoke ]]; then
    "$runner" experiments/smoke.py --worker "$worker" --gateway "$gateway" \
        --output "$out/serving" --telemetry "$out/hardware.jsonl"
else
    rc=0
    "$runner" experiments/telemetry_baseline.py --worker "$worker" --gateway "$gateway" --output "$out/baseline" || rc=$?
    if [[ -s $out/baseline/serving/cells.jsonl ]]; then
        python3 experiments/summarize.py "$out/baseline/serving" \
            --telemetry "$out/baseline/vllm-metrics.jsonl" "$out/hardware.jsonl"
    fi
    exit "$rc"
fi
