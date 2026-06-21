$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating local virtual environment..."
    python -m venv --system-site-packages .venv
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:MPLCONFIGDIR = Join-Path $env:TEMP "usv_faults_matplotlib_cache"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

$RunRoot = Join-Path $ProjectRoot "runs\reports\objective_7_cwru_long10"
$ConfigRoot = Join-Path $RunRoot "configs"
New-Item -ItemType Directory -Force -Path $ConfigRoot | Out-Null

function Invoke-Checked {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $args"
    }
}

function To-YamlPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path -replace "\\", "/"
}

function Write-LongSourceConfig {
    param([Parameter(Mandatory = $true)][string]$Path)

    $CacheDir = To-YamlPath (Join-Path $ProjectRoot "data\external\cwru")
    $Mat97 = To-YamlPath (Join-Path $CacheDir "97.mat")
    $Mat105 = To-YamlPath (Join-Path $CacheDir "105.mat")
    $Mat118 = To-YamlPath (Join-Path $CacheDir "118.mat")
    $Mat130 = To-YamlPath (Join-Path $CacheDir "130.mat")

    @"
attachment_id: public_cwru_long10
source_type: external_cwru
schema_version: 0.1.0
source_url: https://zenodo.org/records/10986655
source_notes: >
  10-second CWRU variant for testing event stability with more windows.
cache_dir: $CacheDir
default_duration_s: 10.0
sampling:
  raw_sample_rate_hz: 12000
channel_profile:
  name: cwru_drive_end_vibration_12khz
  expected_input_dim: 1200
  vibration_channels:
    - drive_end_vibration
  current_channels: []
  scalar_channels: []
  scalar_features: []
trials:
  - trial_id: 2026-05-17_CWRU_long10_normal_train_97
    path: $Mat97
    file_name: 97.mat
    mat_variable: X097_DE_time
    start_sample: 0
    duration_s: 10.0
    baseline_id: 0
    baseline_name: cwru_normal_0hp_1797rpm
    fault_label: none
    fault_induced: false
    rpm: 1797
    load_hp: 0
  - trial_id: 2026-05-17_CWRU_long10_normal_validation_97
    path: $Mat97
    file_name: 97.mat
    mat_variable: X097_DE_time
    start_sample: 120000
    duration_s: 10.0
    baseline_id: 0
    baseline_name: cwru_normal_0hp_1797rpm
    fault_label: none
    fault_induced: false
    rpm: 1797
    load_hp: 0
  - trial_id: 2026-05-17_CWRU_long10_inner_fault_105
    path: $Mat105
    file_name: 105.mat
    mat_variable: X105_DE_time
    start_sample: 0
    duration_s: 10.0
    baseline_id: 0
    baseline_name: cwru_fault_0hp_1797rpm
    fault_label: inner_race_fault_007
    fault_induced: true
    rpm: 1797
    load_hp: 0
  - trial_id: 2026-05-17_CWRU_long10_ball_fault_118
    path: $Mat118
    file_name: 118.mat
    mat_variable: X118_DE_time
    start_sample: 0
    duration_s: 10.0
    baseline_id: 0
    baseline_name: cwru_fault_0hp_1797rpm
    fault_label: ball_fault_007
    fault_induced: true
    rpm: 1797
    load_hp: 0
  - trial_id: 2026-05-17_CWRU_long10_outer_fault_130
    path: $Mat130
    file_name: 130.mat
    mat_variable: X130_DE_time
    start_sample: 0
    duration_s: 10.0
    baseline_id: 0
    baseline_name: cwru_fault_0hp_1797rpm
    fault_label: outer_race_fault_007
    fault_induced: true
    rpm: 1797
    load_hp: 0
"@ | Set-Content -Encoding UTF8 $Path
}

function Write-LongDatasetConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RawRoot
    )

    $RawRootYaml = To-YamlPath $RawRoot
    @"
dataset_id: ds_public_cwru_long10
source_type: external_cwru
raw_trial_root: $RawRootYaml
source_trials:
  - 2026-05-17_CWRU_long10_normal_train_97
  - 2026-05-17_CWRU_long10_normal_validation_97
  - 2026-05-17_CWRU_long10_inner_fault_105
  - 2026-05-17_CWRU_long10_ball_fault_118
  - 2026-05-17_CWRU_long10_outer_fault_130
windowing:
  window_ms: 100
  stride_ms: 100
preprocessing:
  vibration_sample_rate_hz: 12000
  current_sample_rate_hz: 12000
  scalar_features: []
  expected_input_dim: 1200
scaling:
  method: standard
  fit_on: healthy_train_only
  channel_profile: cwru_drive_end_vibration_12khz
  reduced_profile_reason: CWRU adapter is vibration-only and is not padded to the 2109-D USV profile.
split:
  strategy: by_trial
  train:
    - 2026-05-17_CWRU_long10_normal_train_97
  validation:
    - 2026-05-17_CWRU_long10_normal_validation_97
  test:
    - 2026-05-17_CWRU_long10_inner_fault_105
    - 2026-05-17_CWRU_long10_ball_fault_118
    - 2026-05-17_CWRU_long10_outer_fault_130
"@ | Set-Content -Encoding UTF8 $Path
}

function Write-Summary {
    param(
        [Parameter(Mandatory = $true)][string]$ReportDir,
        [Parameter(Mandatory = $true)][string]$OutPath
    )

    $detection = Import-Csv (Join-Path $ReportDir "poc_detection_metrics.csv") | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
    $isolation = Import-Csv (Join-Path $ReportDir "poc_isolation_metrics.csv")
    $events = Import-Csv (Join-Path $ReportDir "poc_event_metrics.csv")
    $windowOverall = $isolation | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
    $windowBall = $isolation | Where-Object { $_.fault_label -eq "ball_fault_007" } | Select-Object -First 1
    $windowInner = $isolation | Where-Object { $_.fault_label -eq "inner_race_fault_007" } | Select-Object -First 1
    $windowOuter = $isolation | Where-Object { $_.fault_label -eq "outer_race_fault_007" } | Select-Object -First 1
    $eventOverall = $events | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
    $eventBall = $events | Where-Object { $_.fault_label -eq "ball_fault_007" } | Select-Object -First 1
    $eventInner = $events | Where-Object { $_.fault_label -eq "inner_race_fault_007" } | Select-Object -First 1
    $eventOuter = $events | Where-Object { $_.fault_label -eq "outer_race_fault_007" } | Select-Object -First 1

    $summary = [pscustomobject]@{
        variant = "cwru_long10"
        window_count = $detection.window_count
        anomaly_count = $detection.anomaly_count
        false_positive_rate = $detection.false_positive_rate
        true_fault_detection_rate = $detection.true_fault_detection_rate
        window_true_fault_isolation_rate = $windowOverall.true_fault_isolation_rate
        window_ball_isolation_rate = $windowBall.true_fault_isolation_rate
        window_inner_isolation_rate = $windowInner.true_fault_isolation_rate
        window_outer_novel_rate = $windowOuter.withheld_novel_rate
        event_false_positive_rate = $eventOverall.event_false_positive_rate
        event_fault_detection_rate = $eventOverall.event_fault_detection_rate
        event_true_fault_isolation_rate = $eventOverall.event_true_fault_isolation_rate
        event_ball_isolation_rate = $eventBall.event_true_fault_isolation_rate
        event_inner_isolation_rate = $eventInner.event_true_fault_isolation_rate
        event_outer_novel_rate = $eventOuter.event_withheld_novel_rate
        event_fault_latency_s = $eventOverall.event_fault_latency_s
        report_dir = $ReportDir
    }
    $summary | Export-Csv -NoTypeInformation -Path $OutPath
    $summary | Format-List
}

$SourceConfig = Join-Path $ConfigRoot "public_cwru_long10.yaml"
$DatasetConfig = Join-Path $ConfigRoot "dataset_public_cwru_long10.yaml"
$RawRoot = Join-Path $ProjectRoot "data\raw\public_cwru_long10"
$DatasetDir = "data/processed/datasets/ds_public_cwru_long10"
$ModelDir = "artifacts/models/run_public_cwru_long10_sdae"
$DictionaryDir = "artifacts/dictionaries/dict_public_cwru_long10"
$ReportDir = "runs/reports/objective_7_cwru_long10"

Write-LongSourceConfig -Path $SourceConfig
Write-LongDatasetConfig -Path $DatasetConfig -RawRoot $RawRoot

Invoke-Checked -m usv_faults.cli attach-data --source cwru --config $SourceConfig --out $RawRoot
Invoke-Checked -m usv_faults.cli qc --trial data/raw/public_cwru_long10/2026-05-17_CWRU_long10_normal_train_97
Invoke-Checked -m usv_faults.cli make-dataset --config $DatasetConfig --out $DatasetDir
Invoke-Checked -m usv_faults.cli train-sdae --dataset $DatasetDir --config configs/baseline_sdae_public_cwru.yaml --out $ModelDir
Invoke-Checked -m usv_faults.cli build-dictionary --model $ModelDir --dataset $DatasetDir --config configs/hdbscan_public_cwru.yaml --out $DictionaryDir
Invoke-Checked -m usv_faults.cli evaluate --model $ModelDir --dictionary $DictionaryDir --dataset $DatasetDir --out $ReportDir
Write-Summary -ReportDir $ReportDir -OutPath (Join-Path $RunRoot "summary.csv")
