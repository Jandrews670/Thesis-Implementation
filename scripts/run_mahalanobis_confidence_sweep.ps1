param(
    [double[]]$Confidences = @(0.60, 0.80, 0.90, 0.95, 0.99),
    [int]$MetricWarmupWindows = 10,
    [switch]$EmpiricalEnabled,
    [double]$EmpiricalPercentile = 0.74,
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

$EmpiricalPercentileText = $EmpiricalPercentile.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
$EmpiricalPercentileId = $EmpiricalPercentileText.Replace(".", "")
$UseEmpirical = [bool]$EmpiricalEnabled
$EmpiricalTag = if ($UseEmpirical) {
    "empirical_p$EmpiricalPercentileId"
} else {
    ""
}
$SweepRoot = Join-Path $ProjectRoot "runs\reports\mahalanobis_confidence_sweep"
if ($EmpiricalTag) {
    $SweepRoot = Join-Path $SweepRoot $EmpiricalTag
}
$ConfigRoot = Join-Path $SweepRoot "configs"
New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ConfigRoot | Out-Null

$Datasets = @(
    @{
        Id = "public_cwru"
        BaseConfig = "configs/hdbscan_public_cwru.yaml"
        Model = "artifacts/models/run_public_cwru_sdae_objective_7"
        Dataset = "data/processed/datasets/ds_public_cwru_objective_7"
    },
    @{
        Id = "public_ims"
        BaseConfig = "configs/hdbscan_public_ims.yaml"
        Model = "artifacts/models/run_public_ims_sdae_expanded"
        Dataset = "data/processed/datasets/ds_public_ims_expanded"
    },
    @{
        Id = "public_femto"
        BaseConfig = "configs/hdbscan_public_femto.yaml"
        Model = "artifacts/models/run_public_femto_sdae"
        Dataset = "data/processed/datasets/ds_public_femto"
    }
)

function Invoke-Checked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Write-Host "$Python -B $($Args -join ' ')"
    & $Python -B @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Set-Or-AppendYamlScalar {
    param(
        [string]$Text,
        [string]$Key,
        [string]$Value
    )
    $pattern = "(?m)^$([regex]::Escape($Key)):\s*.*$"
    $line = "$Key`: $Value"
    if ($Text -match $pattern) {
        return ($Text -replace $pattern, $line)
    }
    return "$Text`n$line`n"
}

function Get-OverallRow {
    param([string]$CsvPath)
    return Import-Csv $CsvPath | Where-Object { $_.scope -eq "overall" } | Select-Object -First 1
}

$Rows = New-Object System.Collections.Generic.List[object]

foreach ($dataset in $Datasets) {
    foreach ($confidence in $Confidences) {
        $confidenceText = $confidence.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
        $confidenceId = $confidenceText.Replace(".", "")
        $runSuffix = if ($UseEmpirical) { $EmpiricalTag } else { "no_empirical" }
        $runId = "$($dataset.Id)_chi_c$($confidenceId)_$runSuffix"
        $configPath = Join-Path $ConfigRoot "$runId.yaml"
        $dictionaryDir = "artifacts/dictionaries/sweeps/dict_$runId"
        $reportRoot = if ($EmpiricalTag) { "runs/reports/mahalanobis_confidence_sweep/$EmpiricalTag" } else { "runs/reports/mahalanobis_confidence_sweep" }
        $reportDir = "$reportRoot/$runId"

        $configText = Get-Content -Path $dataset.BaseConfig -Raw
        $configText = Set-Or-AppendYamlScalar -Text $configText -Key "mahalanobis_confidence" -Value $confidenceText
        $configText = Set-Or-AppendYamlScalar -Text $configText -Key "mahalanobis_empirical_enabled" -Value ([string]$UseEmpirical).ToLowerInvariant()
        if ($UseEmpirical) {
            $configText = Set-Or-AppendYamlScalar -Text $configText -Key "mahalanobis_empirical_percentile" -Value $EmpiricalPercentileText
        }
        Set-Content -Path $configPath -Value $configText -Encoding UTF8

        Write-Host ""
        Write-Host "=== $runId ==="
        Invoke-Checked -m usv_faults.cli build-dictionary `
            --model $dataset.Model `
            --dataset $dataset.Dataset `
            --config $configPath `
            --out $dictionaryDir

        Invoke-Checked -m usv_faults.cli evaluate `
            --model $dataset.Model `
            --dictionary $dictionaryDir `
            --dataset $dataset.Dataset `
            --out $reportDir `
            --metric-warmup-windows $MetricWarmupWindows

        $dictionaryJson = Get-Content -Path (Join-Path $dictionaryDir "dictionary.json") -Raw | ConvertFrom-Json
        $detection = Get-OverallRow (Join-Path $reportDir "poc_detection_metrics.csv")
        $isolation = Get-OverallRow (Join-Path $reportDir "poc_isolation_metrics.csv")
        $event = Get-OverallRow (Join-Path $reportDir "poc_event_metrics.csv")

        $Rows.Add([pscustomobject]@{
            dataset = $dataset.Id
            mahalanobis_confidence = $confidenceText
            empirical_enabled = $UseEmpirical
            empirical_percentile = if ($UseEmpirical) { $EmpiricalPercentileText } else { "" }
            metric_warmup_windows = $MetricWarmupWindows
            chi_square_threshold = $dictionaryJson.mahalanobis.threshold
            dictionary_entry_count = @($dictionaryJson.entries).Count
            dictionary_dir = $dictionaryDir
            report_dir = $reportDir
            window_count = $detection.window_count
            anomaly_count = $detection.anomaly_count
            window_false_positive_rate = $detection.false_positive_rate
            window_fault_detection_rate = $detection.true_fault_detection_rate
            window_true_fault_isolation_rate = $isolation.true_fault_isolation_rate
            window_withheld_novel_rate = $isolation.withheld_novel_rate
            event_false_positive_rate = $event.event_false_positive_rate
            event_fault_detection_rate = $event.event_fault_detection_rate
            event_true_fault_isolation_rate = $event.event_true_fault_isolation_rate
            event_withheld_novel_rate = $event.event_withheld_novel_rate
        })
    }
}

$summaryPath = Join-Path $SweepRoot "summary.csv"
$Rows | Export-Csv -Path $summaryPath -NoTypeInformation

Write-Host ""
Write-Host "Summary written to $summaryPath"
$Rows |
    Sort-Object dataset, mahalanobis_confidence |
    Format-Table dataset, mahalanobis_confidence, chi_square_threshold, window_fault_detection_rate, window_true_fault_isolation_rate, window_withheld_novel_rate, event_fault_detection_rate, event_true_fault_isolation_rate, event_withheld_novel_rate -AutoSize
