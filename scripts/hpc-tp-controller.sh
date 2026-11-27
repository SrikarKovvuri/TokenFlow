#!/bin/bash -l
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --job-name=tf-tp-control
set -euo pipefail
umask 077
root=${SLURM_SUBMIT_DIR:?}
tp=${1:?expected TP size}
cap=${2:?expected sequence limit}
export PYTHONPATH="$root/src"
export PIP_CACHE_DIR="$TMPDIR/pip"
out="$root/results/$SLURM_JOB_ID"
mkdir -p "$root/results"
mkdir "$out"
cd "$out"
cleanup() {
    trap - TERM INT EXIT
    if [[ -s prefix ]]; then
        while read -r id name; do
            if [[ "$id" =~ ^[0-9]+$ && "$name" == "$(cat prefix)"* ]]; then
                scancel --name="$name" "$id" || true
            fi
        done < <(squeue --local --me -h -o '%i %j')
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
python3.12 -m venv "$TMPDIR/gateway"
py="$TMPDIR/gateway/bin/python"
"$py" -m pip install --disable-pip-version-check -q 'pydantic-settings>=2.6,<3' 'fastapi>=0.115,<1' 'httpx>=0.27,<1' 'uvicorn>=0.32,<1'
if ! "$py" -c 'import sqlite3' 2>/dev/null; then
    "$py" -m pip install --disable-pip-version-check -q pysqlite3-binary
    mkdir "$TMPDIR/sqlite-shim"
    printf '%s\n' 'import sys, pysqlite3' 'sys.modules["sqlite3"] = pysqlite3' > "$TMPDIR/sqlite-shim/sitecustomize.py"
    export PYTHONPATH="$TMPDIR/sqlite-shim:$PYTHONPATH"
fi
"$py" -m pip freeze > gateway-packages.txt
"$py" -u "$root/experiments/hpc_tp_controller.py" --tp "$tp" --cap "$cap"
