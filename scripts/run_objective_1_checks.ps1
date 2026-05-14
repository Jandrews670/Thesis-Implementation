$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating local virtual environment..."
    python -m venv --system-site-packages .venv
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"

& $Python -m unittest discover -s tests
& $Python -m usv_faults.cli --help
& $Python -m usv_faults.cli attach-data --source synthetic --config configs/poc_synthetic_smoke.yaml --out data/raw/trials_smoke
& $Python -m usv_faults.cli qc --trial data/raw/trials_smoke/2026-05-14_POC_B0_nominal_T001
