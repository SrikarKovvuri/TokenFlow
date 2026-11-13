#!/usr/bin/env bash
# Invoke through sbatch from an extracted snapshot.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
mode=${1:-smoke}
case "$mode" in smoke|baseline) ;; *) echo "Usage: sbatch scripts/hpc-job.sh [smoke|baseline]" >&2; exit 2;; esac
root=$(pwd -P)
image=${TOKENFLOW_SIF:-"$HOME/vllm-runtime/containers/vllm.sif"}
model=${TOKENFLOW_MODEL_DIR:-"$HOME/vllm-runtime/models/qwen2.5-14b-instruct"}
out=${TOKENFLOW_RESULTS_DIR:-"$HOME/vllm-runtime/results"}/local-$SLURM_JOB_ID
test -r "$image"
test -r "$model/config.json"
test -r "$root/scripts/hpc-run.sh"
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(realpath "$out")
printf 'Job %s on %s; output %s\n' "$SLURM_JOB_ID" "$(hostname)" "$out"
module load apptainer/1.4.5
# One SIF extraction for all processes. Keep temporary data off shared home storage.
export APPTAINER_TMPDIR=${TOKENFLOW_APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}
df -h "$APPTAINER_TMPDIR"
exec apptainer exec --nv --bind "$root:/tokenflow:ro,$model:/models/qwen14b:ro,$out:/run-output" \
    --pwd /tokenflow "$image" bash /tokenflow/scripts/hpc-run.sh "$mode"
