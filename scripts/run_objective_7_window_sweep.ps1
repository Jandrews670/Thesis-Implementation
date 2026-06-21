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

$SweepRoot = Join-Path $ProjectRoot "runs\reports\objective_7_window_sweep"
$ConfigRoot = Join-Path $SweepRoot "configs"
New-Item -ItemType Directory -Force -Path $ConfigRoot | Out-Null

function Invoke-Checked {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Python $args"
    }
}

function Write-HdbscanConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$RollingWindowSize,
        [Parameter(Mandatory = $true)][int]$MinRuntimeClusterSize,
        [Parameter(Mandatory = $true)][int]$MinClusterSize,
        [Parameter(Mandatory = $true)][int]$MinSamples,
        [Parameter(Mandatory = $true)][double]$ClusterMatchMinMemberFraction
    )

    @"
rolling_window_size: $RollingWindowSize
min_cluster_size: $MinClusterSize
min_samples: $MinSamples
metric: euclidean
cluster_selection_method: eom
allow_single_cluster: false
mahalanobis_confidence: 0.99
min_runtime_cluster_size: $MinRuntimeClusterSize
cluster_match_min_member_fraction: $ClusterMatchMinMemberFraction
event_window_size: 30
event_min_anomaly_votes: 3
event_min_anomaly_fraction: 0.30
event_min_known_votes: 3
event_min_known_fraction: 0.15
event_min_known_purity: 0.50
event_min_novel_votes: 3
event_min_novel_fraction: 0.15
dictionary_baseline_id: 0
known_fault_labels:
  - inner_race_fault_007
  - ball_fault_007
withheld_fault_labels:
  - outer_race_fault_007
"@ | Set-Content -Encoding UTF8 $Path
}

function Write-CwruDatasetConfig200ms {
    param([Parameter(Mandatory = $true)][string]$Path)

    $RawTrialRoot = (Join-Path $ProjectRoot "data\raw\public_cwru") -replace "\\", "/"

    @"
dataset_id: ds_public_cwru_200ms_latent32
source_type: external_cwru
raw_trial_root: $RawTrialRoot
source_trials:
  - 2026-05-17_CWRU_normal_train_97
  - 2026-05-17_CWRU_normal_validation_97
  - 2026-05-17_CWRU_inner_fault_105
  - 2026-05-17_CWRU_ball_fault_118
  - 2026-05-17_CWRU_outer_fault_130
windowing:
  window_ms: 200
  stride_ms: 100
preprocessing:
  vibration_sample_rate_hz: 12000
  current_sample_rate_hz: 12000
  scalar_features: []
  expected_input_dim: 2400
scaling:
  method: standard
  fit_on: healthy_train_only
  channel_profile: cwru_drive_end_vibration_12khz
  reduced_profile_reason: CWRU adapter is vibration-only and is not padded to the 2109-D USV profile.
split:
  strategy: by_trial
  train:
    - 2026-05-17_CWRU_normal_train_97
  validation:
    - 2026-05-17_CWRU_normal_validation_97
  test:
    - 2026-05-17_CWRU_inner_fault_105
    - 2026-05-17_CWRU_ball_fault_118
    - 2026-05-17_CWRU_outer_fault_130
"@ | Set-Content -Encoding UTF8 $Path
}

function Write-SdaeConfig200ms {
    param([Parameter(Mandatory = $true)][string]$Path)

    @"
model:
  input_dim: 2400
  hidden_dims: [256, 128]
  latent_dim: 32
  hidden_activation: relu
  output_activation: sigmoid
  masking_noise: 0.10
  l1_lambda: 0.0001
training:
  optimizer: adam
  learning_rate: 0.001
  batch_size: 16
  epochs: 8
  early_stopping: false
  seed: 20260517
threshold:
  method: validation_percentile
  target_false_positive_rate: 0.02
"@ | Set-Content -Encoding UTF8 $Path
}

function Count-Decision {
    param(
        [Parameter(Mandatory = $true)]$Rows,
        [Parameter(Mandatory = $true)][string]$FaultLabel,
        [Parameter(Mandatory = $true)][string]$Decision
    )
    return @($Rows | Where-Object {
        $_.fault_label -eq $FaultLabel -and $_.is_anomaly -eq "True" -and $_.dictionary_decision -eq $Decision
    }).Count
}

function Count-EventDecision {
    param(
        [Parameter(Mandatory = $true)]$Rows,
        [Parameter(Mandatory = $true)][string]$FaultLabel,
        [Parameter(Mandatory = $true)][string]$Decision
    )
    return @($Rows | Where-Object {
        $_.fault_label -eq $FaultLabel -and $_.is_fault -eq "True" -and $_.event_decision -eq $Decision
    }).Count
}

function Read-VariantSummary {
    param(
        [Parameter(Mandatory = $true)][string]$VariantId,
        [Parameter(Mandatory = $true)][string]$DatasetDir,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$DictionaryDir,
        [Parameter(Mandatory = $true)][string]$ReportDir,
        [Parameter(Mandatory = $true)][int]$InputWindowMs,
        [Parameter(Mandatory = $true)][int]$InputDim,
        [Parameter(Mandatory = $true)][int]$LatentDim,
        [Parameter(Mandatory = $true)][int]$RollingWindowSize,
        [Parameter(Mandatory = $true)][int]$MinRuntimeClusterSize,
        [Parameter(Mandatory = $true)][double]$ClusterMatchMinMemberFraction
    )

    $dictionary = Get-Content (Join-Path $DictionaryDir "dictionary.json") -Raw | ConvertFrom-Json
    $detection = Import-Csv (Join-Path $ReportDir "poc_detection_metrics.csv") | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
    $isolation = Import-Csv (Join-Path $ReportDir "poc_isolation_metrics.csv")
    $eventMetrics = Import-Csv (Join-Path $ReportDir "poc_event_metrics.csv")
    $overall = $isolation | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
    $ball = $isolation | Where-Object { $_.fault_label -eq "ball_fault_007" } | Select-Object -First 1
    $inner = $isolation | Where-Object { $_.fault_label -eq "inner_race_fault_007" } | Select-Object -First 1
    $outer = $isolation | Where-Object { $_.fault_label -eq "outer_race_fault_007" } | Select-Object -First 1
    $eventOverall = $eventMetrics | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
    $eventBall = $eventMetrics | Where-Object { $_.fault_label -eq "ball_fault_007" } | Select-Object -First 1
    $eventInner = $eventMetrics | Where-Object { $_.fault_label -eq "inner_race_fault_007" } | Select-Object -First 1
    $eventOuter = $eventMetrics | Where-Object { $_.fault_label -eq "outer_race_fault_007" } | Select-Object -First 1
    $decisions = Import-Csv (Join-Path $ReportDir "poc_window_decisions.csv")
    $eventDecisions = Import-Csv (Join-Path $ReportDir "poc_event_decisions.csv")

    [pscustomobject]@{
        variant = $VariantId
        input_window_ms = $InputWindowMs
        input_dim = $InputDim
        latent_dim = $LatentDim
        rolling_window_size = $RollingWindowSize
        min_runtime_cluster_size = $MinRuntimeClusterSize
        cluster_match_min_member_fraction = $ClusterMatchMinMemberFraction
        dictionary_entries = $dictionary.entries.Count
        window_count = $detection.window_count
        anomaly_count = $detection.anomaly_count
        false_positive_rate = $detection.false_positive_rate
        true_fault_detection_rate = $detection.true_fault_detection_rate
        true_fault_isolation_rate = $overall.true_fault_isolation_rate
        withheld_novel_rate = $overall.withheld_novel_rate
        fault_isolation_latency_s = $overall.fault_isolation_latency_s
        dbcv_score = $overall.dbcv_score
        ball_isolation_rate = $ball.true_fault_isolation_rate
        inner_isolation_rate = $inner.true_fault_isolation_rate
        outer_novel_rate = $outer.withheld_novel_rate
        event_false_positive_rate = $eventOverall.event_false_positive_rate
        event_fault_detection_rate = $eventOverall.event_fault_detection_rate
        event_true_fault_isolation_rate = $eventOverall.event_true_fault_isolation_rate
        event_withheld_novel_rate = $eventOverall.event_withheld_novel_rate
        event_fault_latency_s = $eventOverall.event_fault_latency_s
        event_ball_isolation_rate = $eventBall.event_true_fault_isolation_rate
        event_inner_isolation_rate = $eventInner.event_true_fault_isolation_rate
        event_outer_novel_rate = $eventOuter.event_withheld_novel_rate
        ball_known = Count-Decision $decisions "ball_fault_007" "known"
        ball_noise = Count-Decision $decisions "ball_fault_007" "novel_cluster_noise"
        inner_known = Count-Decision $decisions "inner_race_fault_007" "known"
        inner_novel = Count-Decision $decisions "inner_race_fault_007" "novel"
        inner_noise = Count-Decision $decisions "inner_race_fault_007" "novel_cluster_noise"
        outer_novel = Count-Decision $decisions "outer_race_fault_007" "novel"
        outer_noise = Count-Decision $decisions "outer_race_fault_007" "novel_cluster_noise"
        event_ball_known = Count-EventDecision $eventDecisions "ball_fault_007" "known"
        event_ball_novel = Count-EventDecision $eventDecisions "ball_fault_007" "novel"
        event_inner_known = Count-EventDecision $eventDecisions "inner_race_fault_007" "known"
        event_inner_novel = Count-EventDecision $eventDecisions "inner_race_fault_007" "novel"
        event_outer_novel = Count-EventDecision $eventDecisions "outer_race_fault_007" "novel"
        dataset_dir = $DatasetDir
        model_dir = $ModelDir
        dictionary_dir = $DictionaryDir
        report_dir = $ReportDir
    }
}

$RawNormalTrial = Join-Path $ProjectRoot "data\raw\public_cwru\2026-05-17_CWRU_normal_train_97"
if (-not (Test-Path $RawNormalTrial)) {
    Invoke-Checked -m usv_faults.cli attach-data --source cwru --config configs/public_cwru.yaml --out data/raw/public_cwru
} else {
    Write-Host "Using existing CWRU raw trials under data\raw\public_cwru"
}

Invoke-Checked -m usv_faults.cli qc --trial data/raw/public_cwru/2026-05-17_CWRU_normal_train_97

$BaseDataset = "data/processed/datasets/ds_public_cwru_objective_7"
$BaseModel = "artifacts/models/run_public_cwru_sdae_objective_7"
Invoke-Checked -m usv_faults.cli make-dataset --config configs/dataset_public_cwru.yaml --out $BaseDataset
Invoke-Checked -m usv_faults.cli train-sdae --dataset $BaseDataset --config configs/baseline_sdae_public_cwru.yaml --out $BaseModel

$summaries = @()
$rollingVariants = @(
    @{ id = "roll30_min3_frac0"; rolling = 30; min_runtime = 3; min_cluster = 3; min_samples = 1; fraction = 0.0 },
    @{ id = "roll60_min3_frac0"; rolling = 60; min_runtime = 3; min_cluster = 3; min_samples = 1; fraction = 0.0 },
    @{ id = "roll60_min5_frac0"; rolling = 60; min_runtime = 5; min_cluster = 3; min_samples = 1; fraction = 0.0 },
    @{ id = "roll90_min5_frac0"; rolling = 90; min_runtime = 5; min_cluster = 3; min_samples = 1; fraction = 0.0 },
    @{ id = "roll60_min3_frac50"; rolling = 60; min_runtime = 3; min_cluster = 3; min_samples = 1; fraction = 0.5 }
)

foreach ($variant in $rollingVariants) {
    $configPath = Join-Path $ConfigRoot "$($variant.id)_hdbscan.yaml"
    Write-HdbscanConfig `
        -Path $configPath `
        -RollingWindowSize $variant.rolling `
        -MinRuntimeClusterSize $variant.min_runtime `
        -MinClusterSize $variant.min_cluster `
        -MinSamples $variant.min_samples `
        -ClusterMatchMinMemberFraction $variant.fraction

    $dictDir = "artifacts/dictionaries/dict_public_cwru_$($variant.id)"
    $reportDir = "runs/reports/objective_7_window_sweep/$($variant.id)"
    Invoke-Checked -m usv_faults.cli build-dictionary --model $BaseModel --dataset $BaseDataset --config $configPath --out $dictDir
    Invoke-Checked -m usv_faults.cli evaluate --model $BaseModel --dictionary $dictDir --dataset $BaseDataset --out $reportDir

    $summaries += Read-VariantSummary `
        -VariantId $variant.id `
        -DatasetDir $BaseDataset `
        -ModelDir $BaseModel `
        -DictionaryDir $dictDir `
        -ReportDir $reportDir `
        -InputWindowMs 100 `
        -InputDim 1200 `
        -LatentDim 16 `
        -RollingWindowSize $variant.rolling `
        -MinRuntimeClusterSize $variant.min_runtime `
        -ClusterMatchMinMemberFraction $variant.fraction
}

$Dataset200Config = Join-Path $ConfigRoot "dataset_public_cwru_200ms_latent32.yaml"
$Train200Config = Join-Path $ConfigRoot "baseline_sdae_public_cwru_200ms_latent32.yaml"
$Hdbscan200Config = Join-Path $ConfigRoot "input200_roll60_min3_frac0_hdbscan.yaml"
Write-CwruDatasetConfig200ms -Path $Dataset200Config
Write-SdaeConfig200ms -Path $Train200Config
Write-HdbscanConfig `
    -Path $Hdbscan200Config `
    -RollingWindowSize 60 `
    -MinRuntimeClusterSize 3 `
    -MinClusterSize 3 `
    -MinSamples 1 `
    -ClusterMatchMinMemberFraction 0.0

$Dataset200 = "data/processed/datasets/ds_public_cwru_200ms_latent32"
$Model200 = "artifacts/models/run_public_cwru_200ms_latent32"
$Dict200 = "artifacts/dictionaries/dict_public_cwru_input200_roll60_min3_frac0"
$Report200 = "runs/reports/objective_7_window_sweep/input200_roll60_min3_frac0"

Invoke-Checked -m usv_faults.cli make-dataset --config $Dataset200Config --out $Dataset200
Invoke-Checked -m usv_faults.cli train-sdae --dataset $Dataset200 --config $Train200Config --out $Model200
Invoke-Checked -m usv_faults.cli build-dictionary --model $Model200 --dataset $Dataset200 --config $Hdbscan200Config --out $Dict200
Invoke-Checked -m usv_faults.cli evaluate --model $Model200 --dictionary $Dict200 --dataset $Dataset200 --out $Report200

$summaries += Read-VariantSummary `
    -VariantId "input200_roll60_min3_frac0" `
    -DatasetDir $Dataset200 `
    -ModelDir $Model200 `
    -DictionaryDir $Dict200 `
    -ReportDir $Report200 `
    -InputWindowMs 200 `
    -InputDim 2400 `
    -LatentDim 32 `
    -RollingWindowSize 60 `
    -MinRuntimeClusterSize 3 `
    -ClusterMatchMinMemberFraction 0.0

$SummaryPath = Join-Path $SweepRoot "summary.csv"
$summaries | Export-Csv -NoTypeInformation -Path $SummaryPath
$summaries | Format-Table -AutoSize
Write-Host "Wrote sweep summary to $SummaryPath"
