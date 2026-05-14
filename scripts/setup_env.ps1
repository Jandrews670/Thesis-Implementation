$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    python -m venv --system-site-packages .venv
}

$PyvenvConfig = Join-Path $ProjectRoot ".venv\pyvenv.cfg"
if (Test-Path $PyvenvConfig) {
    (Get-Content $PyvenvConfig) -replace "include-system-site-packages = false", "include-system-site-packages = true" | Set-Content $PyvenvConfig
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "Environment ready."
Write-Host "Use: .\.venv\Scripts\python.exe -m usv_faults.cli --help"
Write-Host "This objective runs via PYTHONPATH=src and does not install/download packages."
