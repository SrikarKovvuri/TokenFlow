# Restarting the HPC worker

Recovery status: see the verified recovery section below. The allocation and container version are recovered; the host model directory still needs to be located.

## 1. Reach the login node

Connect to the required university network/VPN, then use the login command from
your existing shell history. In local PowerShell, set the actual login destination:

```powershell
$HpcLogin = "YOUR_USER@LOGIN_HOST"
ssh $HpcLogin
```

If this times out, resolve network reachability before trying allocation or model
startup. Authentication/MFA, if required, is performed in your own terminal.

## 2. Recover the known-working cluster commands

In the logged-in cluster shell, inspect relevant history:

```bash
history | grep -E 'salloc|srun|sbatch|vllm|apptainer|singularity|module (load|use)|activate|qwen' | tail -n 80
grep -E 'salloc|srun|sbatch|vllm|apptainer|singularity|module (load|use)|activate|qwen' ~/.bash_history | tail -n 80
```

Find saved job scripts in the project/scratch directory identified by that history:

```bash
find . -maxdepth 4 -type f \( -name '*.sbatch' -o -name '*.slurm' -o -name '*vllm*.sh' -o -name '*tokenflow*.sh' \)
squeue -u "$(id -un)"
```

Recover and verify these before launching:

- The exact partition/account, GPU request, CPU/RAM request, and wall time.
- Module loads, virtual environment, or container image and bind mounts.
- The model path and complete serving command.
- Whether an existing allocation/server can be reused.

The previous server advertised model root `/models/qwen-0.5b`, served name
`tokenflow-model`, context limit 2048, worker port 8000, and vLLM 0.10.2. Those are
observed runtime facts, not a complete recovered launch command. In particular,
the model path may be inside a container.

## 3. Start the allocation and worker

Run the recovered allocation/environment/serving commands. Capture the new
compute-node hostname and reachable address from the allocated compute shell:

```bash
hostname
hostname -I
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

The previously successful tunnel targeted the compute address through the login
node. That requires the worker to listen on an interface reachable from the login
node; loopback-only binding requires a different SSH route. Check the recovered
serving command instead of assuming the bind address. Keep the allocation and
worker process alive while serving.

## 4. Recreate the tunnel locally

In another local PowerShell terminal, supply the *current* allocated compute
address. Keep this terminal running:

```powershell
$HpcLogin = "YOUR_USER@LOGIN_HOST"
$ComputeAddress = "CURRENT_ALLOCATED_COMPUTE_ADDRESS"
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L "127.0.0.1:18101:${ComputeAddress}:8000" $HpcLogin
```

This is the forwarding pattern that previously worked. A previous attempt to SSH
directly to a compute node through ProxyJump was not the final working path.
No real compute-node address belongs in committed setup scripts.

## 5. Verify the tunnel and TokenFlow

In a separate local PowerShell terminal, from the repository root:

```powershell
curl.exe --fail --max-time 10 http://127.0.0.1:18101/health
curl.exe --fail --max-time 10 http://127.0.0.1:18101/v1/models
python experiments/smoke.py --worker http://127.0.0.1:18101 --gateway http://127.0.0.1:8001
```

The smoke command discovers the served model, starts a temporary local gateway if
needed, registers the endpoint, waits for healthy status, and verifies JSON/SSE
forwarding. It stops a gateway it starts. The tunnel and HPC allocation remain
separate processes. Do not start a large experiment until this check passes.

## Pending recovery

Replace this section with the verified allocation, environment/container, and
launch commands after cluster access is restored. Keep mutable compute-node
addresses as runtime inputs. No Slurm provisioning has been added to TokenFlow.

## Access reference

Columbia's [Insomnia getting-started guide](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145140/Insomnia%3A%2BGetting%2BStarted)
requires an approved campus network or Columbia/CUIMC VPN for off-campus access,
and documents Duo authentication for login nodes. A connection timeout occurs
before authentication; it does not establish that credentials are wrong.

## One-command read-only history recovery

From the repository root in your own local PowerShell terminal:

```powershell
./scripts/recover-hpc-setup.ps1 -Login "YOUR_USER@LOGIN_HOST"
```

Complete password/Duo authentication directly in that terminal. The command reads
relevant remote history, current Slurm jobs, and up to 20 candidate job scripts
under your home directory. It saves a timestamped text file under the ignored
`experiments/results/` directory. It does not request an allocation or execute
any recovered script. Common token/password assignments are redacted from the
saved output. Paths in history can identify scratch/project locations outside
home that need a targeted follow-up search.

## Verified recovery from October 24

The second recovery attempt completed password/Duo authentication successfully.
The `squeue` call failed to locate Slurm configuration in that non-interactive
shell. This does not prove that interactive Slurm access is broken or that no
jobs exist. Start a normal interactive SSH login and check there; do not invent a
`SLURM_CONF` path or controller address.

### Recovered allocation command

Run on the login node in a normal authenticated SSH session:

```bash
squeue -u "$USER"
srun --account=edu --partition=burst --qos=burst_interactive \
  --nodes=1 --ntasks=1 --gres=gpu:A6000:1 \
  --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

This command is present in the recovered history. A four-hour variant also
appears, but its successful acceptance was not established. Do not copy old job
IDs, compute hostnames, or exclusion lists from history. Check existing jobs
before requesting another allocation.

### Recovered container setup

On the allocated compute node:

```bash
module load apptainer/1.4.5
ls -lh "$HOME/vllm-runtime/containers/vllm.sif"
```

The actual `pull-vllm-local.sh` builds vLLM **0.10.2**, matching the recorded serving
engine. A separate older `pull-vllm.sh` points at **v0.30.0-cu129**. Reuse and inspect
the existing SIF; do not rerun an image pull to restart serving.

### Locate the host model directory

The previous `/models/qwen-0.5b` is a container path. The corresponding host bind
source was not recovered. These are the known candidate locations:

```bash
find "$HOME/vllm-runtime/models" \
  "/insomnia001/depts/edu/users/$USER/tokenflow" \
  -maxdepth 4 -type f -name config.json 2>/dev/null
```

Identify the directory containing the actual Qwen 0.5B model configuration and
weights before launching. Do not select an unrelated model merely because it has
a config.json file.

### Reconstructed serving command

`scripts/start-vllm-worker.sh` is a new helper, not a recovered historical script.
It uses the known image location and recorded model alias, context length, prefix
caching, and GPU-memory-utilization settings. It requires an existing Slurm job,
an explicit host model directory, and an image reporting vLLM 0.10.2. It does not
allocate GPUs or pull a container. Additional historical launch flags remain
unknown, so this is not a claim that every original engine option was recovered.

After copying the helper to the cluster, run inside the allocated compute shell:

```bash
module load apptainer/1.4.5
bash /path/to/start-vllm-worker.sh /absolute/host/path/to/qwen-model
```

The helper prints the current job, hostname, and candidate addresses. Use the
current address in the local tunnel instructions above. Confirm `/health` and
`/v1/models`, then run the local smoke command before benchmarking.

If `squeue` still reports a DNS/configuration error in a normal interactive login,
collect `type -a squeue srun`, `printf '%s\n' "${SLURM_CONF:-unset}"`, and
`module list`; diagnose the installed Slurm environment before submitting a job.
