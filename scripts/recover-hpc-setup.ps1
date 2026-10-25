param(
    [Parameter(Mandatory = $true)][string]$Login,
    [string]$OutputPath = (Join-Path $PSScriptRoot ("../experiments/results/hpc-recovery-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".txt"))
)
$ErrorActionPreference = 'Stop'
if ($Login.StartsWith('-') -or $Login -match '\s') { throw 'Supply an SSH destination such as user@login-host.' }
if (Test-Path -LiteralPath $OutputPath) { throw "Output already exists: $OutputPath" }
$payloadPath = Join-Path $PSScriptRoot 'recover-hpc-history.sh'
$payload = (Get-Content -LiteralPath $payloadPath -Raw).Replace("`r`n", "`n")
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($payload))
$remoteCommand = "printf %s $encoded | base64 --decode | bash"
$parent = Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath))
New-Item -ItemType Directory -Force -Path $parent | Out-Null
Write-Host 'Complete SSH password/Duo authentication in this terminal.'
Write-Host 'Recovery is read-only. It does not allocate GPUs or start vLLM.'
& ssh -o ConnectTimeout=15 $Login $remoteCommand | Tee-Object -FilePath $OutputPath
$sshExitCode = $LASTEXITCODE
if ($sshExitCode -ne 0) { throw "SSH recovery failed (exit $sshExitCode). Partial output: $OutputPath" }
Write-Host "Saved recovery output: $([IO.Path]::GetFullPath($OutputPath))"
