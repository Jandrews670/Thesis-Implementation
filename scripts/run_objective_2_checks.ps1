$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating local virtual environment..."
    python -m venv --system-site-packages .venv
}

$PyvenvConfig = Join-Path $ProjectRoot ".venv\pyvenv.cfg"
if (Test-Path $PyvenvConfig) {
    (Get-Content $PyvenvConfig) -replace "include-system-site-packages = false", "include-system-site-packages = true" | Set-Content $PyvenvConfig
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

function Invoke-Checked {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $args"
    }
}

Invoke-Checked -m unittest discover -s tests
Invoke-Checked -m usv_faults.cli attach-data --source synthetic --config configs/poc_synthetic_smoke.yaml --out data/raw/trials_smoke
Invoke-Checked -m usv_faults.cli preview --trial data/raw/trials_smoke/2026-05-14_POC_B0_nominal_T001
Invoke-Checked -m usv_faults.cli make-dataset --config configs/dataset_poc_synthetic_smoke.yaml --out data/processed/datasets/ds_poc_synthetic_smoke
