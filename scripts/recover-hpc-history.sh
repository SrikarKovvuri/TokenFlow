#!/usr/bin/env bash
# Read-only recovery: no job submission, model startup, or environment changes.
set -uo pipefail
redact() {
    sed -E \
      -e 's/([A-Za-z0-9_]*(TOKEN|PASSWORD|SECRET|API_KEY)[A-Za-z0-9_]*=)[^[:space:]]+/\1<redacted>/gI' \
      -e 's/(--(api-key|hf-token)(=|[[:space:]]+))[^[:space:]]+/\1<redacted>/gI' \
      -e 's/hf_[A-Za-z0-9]{10,}/<redacted-hf-token>/g' \
      -e 's/sk-[A-Za-z0-9_-]{15,}/<redacted-api-key>/g'
}
printf '\n=== LOGIN NODE ===\n'
hostname
printf '\n=== CURRENT JOBS ===\n'
if command -v squeue >/dev/null 2>&1; then
    squeue -u "$(id -un)" -o '%.18i %.12P %.30j %.10T %.12M %.8l %.6D %R'
fi
printf '\n=== RELEVANT SHELL HISTORY ===\n'
for history_file in "$HOME/.bash_history" "$HOME/.zsh_history"; do
    if [ -r "$history_file" ]; then
        printf '\nHistory file: %s\n' "$history_file"
        grep -E 'salloc|srun|sbatch|vllm|apptainer|singularity|module (load|use)|activate|qwen' "$history_file" | tail -n 100 | redact
    fi
done
printf '\n=== JOB SCRIPTS IN HOME (first 20, depth <= 5) ===\n'
find "$HOME" -maxdepth 5 \
  \( -name .cache -o -name .conda -o -name .venv -o -name node_modules -o -name .git \) -prune -o \
  -type f \( -name '*.sbatch' -o -name '*.slurm' -o -name '*vllm*.sh' -o -name '*tokenflow*.sh' \) -print0 2>/dev/null |
head -z -n 20 |
while IFS= read -r -d '' script_file; do
    printf '\nScript: %s\n' "$script_file"
    sed -n '1,180p' "$script_file" | redact
done
printf '\n=== RECOVERY FINISHED ===\n'
