[CmdletBinding()]
param([string]$Login)
$ErrorActionPreference = 'Stop'
if ($Login -and $Login -notmatch '^[\w.-]+@[\w.-]+$') { throw 'Use user@login-host.' }
$repo = Split-Path -Parent $PSScriptRoot
$id = 'tokenflow-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$out = Join-Path $repo "experiments/results/$id.tar.gz"
New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
# Explicit inputs: no Git history, results, model files, or local settings.
& tar.exe -czf $out --exclude=__pycache__ --exclude=*.pyc -C $repo pyproject.toml README.md src/tokenflow experiments/baseline.json experiments/suite.json experiments/workload.example.json experiments/requirements.txt experiments/smoke.py experiments/telemetry_baseline.py experiments/suite.py experiments/run.py experiments/summarize.py experiments/monitor.py experiments/vllm_metrics.py scripts/hpc-job.sh scripts/hpc-run.sh
if ($LASTEXITCODE) { throw 'Snapshot creation failed.' }
Write-Host "Snapshot: $out"
if ($Login) {
    & scp.exe $out "${Login}:$id.tar.gz"
    if ($LASTEXITCODE) { throw 'Snapshot upload failed.' }
    Write-Host "Uploaded. On the cluster login node: mkdir $id && tar -xzf $id.tar.gz -C $id && cd $id"
}
