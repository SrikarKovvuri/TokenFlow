#!/usr/bin/env bash
# Reconstructed from recovered runtime paths and observed baseline settings.
# Run only inside an existing GPU Slurm allocation. This script submits no jobs.
set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo 'Run this inside your allocated GPU job, not on the login node.' >&2
    exit 2
fi
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash $0 /absolute/host/model/directory [/absolute/container.sif]" >&2
    exit 2
fi
model_dir=$(realpath -- "$1")
image=${2:-"$HOME/vllm-runtime/containers/vllm.sif"}
if [ ! -r "$model_dir/config.json" ]; then
    echo "No readable model config found at $model_dir/config.json" >&2
    exit 2
fi
if [ ! -r "$image" ]; then
    echo "Existing container image not found: $image" >&2
    exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo 'Load the recovered environment first: module load apptainer/1.4.5' >&2
    exit 2
fi
version=$(apptainer exec "$image" python3 -c 'import vllm; print(vllm.__version__)')
if [ "$version" != '0.10.2' ]; then
    echo "Image reports vLLM $version; the saved baseline used 0.10.2. Check the image before proceeding." >&2
    exit 2
fi
printf 'Job: %s\nNode: %s\nAddresses: %s\nModel: %s\nImage: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(hostname -I)" "$model_dir" "$image"
printf 'Starting tokenflow-model on port 8000; keep this job/session running.\n'
exec apptainer exec --nv \
    --bind "$model_dir:/models/qwen-0.5b:ro" \
    "$image" python3 -m vllm.entrypoints.openai.api_server \
    --model /models/qwen-0.5b \
    --served-model-name tokenflow-model \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching
