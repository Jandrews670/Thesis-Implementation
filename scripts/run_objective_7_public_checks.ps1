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
$env:MPLCONFIGDIR = Join-Path $env:TEMP "usv_faults_matplotlib_cache"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

function Invoke-Checked {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $args"
    }
}

Invoke-Checked -m unittest tests.test_objective_7
Invoke-Checked -m usv_faults.cli attach-data --source cwru --config configs/public_cwru.yaml --out data/raw/public_cwru
Invoke-Checked -m usv_faults.cli qc --trial data/raw/public_cwru/2026-05-17_CWRU_normal_train_97
Invoke-Checked -m usv_faults.cli make-dataset --config configs/dataset_public_cwru.yaml --out data/processed/datasets/ds_public_cwru_objective_7
Invoke-Checked -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_public_cwru_objective_7 --config configs/baseline_sdae_public_cwru.yaml --out artifacts/models/run_public_cwru_sdae_objective_7
Invoke-Checked -m usv_faults.cli build-dictionary --model artifacts/models/run_public_cwru_sdae_objective_7 --dataset data/processed/datasets/ds_public_cwru_objective_7 --config configs/hdbscan_public_cwru.yaml --out artifacts/dictionaries/dict_public_cwru_objective_7
Invoke-Checked -m usv_faults.cli evaluate --model artifacts/models/run_public_cwru_sdae_objective_7 --dictionary artifacts/dictionaries/dict_public_cwru_objective_7 --dataset data/processed/datasets/ds_public_cwru_objective_7 --out runs/reports/objective_7_public_cwru
