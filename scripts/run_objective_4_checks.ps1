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
$env:MPLCONFIGDIR = Join-Path $ProjectRoot "artifacts\matplotlib_cache_objective_4"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

function Invoke-Checked {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $args"
    }
}

Invoke-Checked -m unittest discover -s tests
Invoke-Checked -m usv_faults.cli attach-data --source synthetic --config configs/poc_synthetic_training_smoke.yaml --out data/raw/trials_training_smoke
Invoke-Checked -m usv_faults.cli make-dataset --config configs/dataset_poc_synthetic_training_smoke.yaml --out data/processed/datasets/ds_poc_synthetic_training_smoke
Invoke-Checked -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --config configs/baseline_sdae_smoke.yaml --out artifacts/models/run_poc_sdae_smoke_objective_4
Invoke-Checked -m usv_faults.cli build-dictionary --model artifacts/models/run_poc_sdae_smoke_objective_4 --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --config configs/hdbscan.yaml --out artifacts/dictionaries/dict_poc_b0_smoke_objective_4
