# Cluster-local TokenFlow

Each run uses a snapshot of the current source, including uncommitted edits.
vLLM, TokenFlow, the monitor and the load generator run inside one Slurm job
and one Apptainer invocation. Services bind to loopback; no tunnel is needed.

From local PowerShell in the repository, connect to VPN and run:

```powershell
.\scripts\package-hpc.ps1 -Login srk2211@insomnia.rcs.columbia.edu
```

Complete Duo. The command prints the exact extraction command. SSH to the
login node and run that command, then submit a smoke job from the snapshot:

```bash
sbatch --account=edu --partition=burst --qos=burst \
  --gres=gpu:A6000:1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
  scripts/hpc-job.sh smoke
```

Check `squeue -u "$USER"` and `tail -f slurm-JOBID.out`. The resource
request is a starting point; cluster acceptance and dependencies have to be
verified by the first job. The existing SIF must provide Python 3.11+.
If imports are missing, the job installs benchmark dependencies into a
temporary virtual environment (requires package index access; see
`dependencies.log`). The existing 14B weights are reused.
If SIF extraction fails for space, inspect the job's local filesystem and
set `TOKENFLOW_APPTAINER_TMPDIR` to a suitable writable location.

Once Slurm accepts the job, you can disconnect. Retrieve results afterward
from local PowerShell (replace JOBID):

```powershell
scp -r srk2211@insomnia.rcs.columbia.edu:vllm-runtime/results/local-JOBID experiments/results/
```

Inspect `exit-code.txt`, `serving/summary.csv`, and the logs. Once smoke
passes, submit again with `baseline` instead of `smoke` and a longer wall
time, e.g. `02:00:00` if your allocation permits it. The baseline preserves
engine scrapes, hardware samples, and a timestamp-joined summary. Compare
against a new cluster-local baseline because the request path has changed.
