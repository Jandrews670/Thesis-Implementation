# User Guide: Configure and Train a Proof-of-Concept SDAE Model

This guide explains how to configure synthetic proof-of-concept data, generate processed windows, and train a Sparse Denoising Autoencoder (SDAE) model using the current implementation.

The current implementation supports:

- synthetic telemetry generation
- raw trial quality checks
- telemetry preview CSVs
- 100 ms window generation
- SDAE training on healthy windows
- reconstruction-threshold selection
- training artifacts and basic plots

It does not yet support HDBSCAN fault dictionaries, FedRep, DANN, external public datasets, or live Teensy/Raspberry Pi collection. Those are later milestones.

## 1. Environment Setup

Run all commands from:

```powershell
C:\Users\jacks\OneDrive\Thesis\Implementation
```

Create or refresh the local environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

The project currently runs through the local `.venv` and `PYTHONPATH=src`. The setup script does not download packages; it reuses packages already installed on this machine through `--system-site-packages`.

For manual commands in PowerShell, set:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "src"
```

Then call the CLI as:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli --help
```

## 2. Choose the Config Set

There are two useful config paths.

Use the smoke configs for fast checks:

```text
configs/poc_synthetic_training_smoke.yaml
configs/dataset_poc_synthetic_training_smoke.yaml
configs/baseline_sdae_smoke.yaml
```

Use the fuller proof-of-concept configs when you intentionally want larger data and the thesis-default model shape:

```text
configs/poc_synthetic.yaml
configs/dataset_poc_synthetic.yaml
configs/baseline_sdae.yaml
```

The smoke configs are intended for quick validation. The full configs are closer to the thesis plan but will take longer and create more data.

## 3. Configure Synthetic Trials

Synthetic trial configuration is controlled by `poc_synthetic*.yaml`.

The important sections are:

```yaml
sampling:
  raw_sample_rate_hz: 10000
  window_ms: 100
  stride_ms: 100
  current_decimated_hz: 1000

channel_profile:
  expected_input_dim: 2109
  vibration_channels:
    - motor_vibration
    - rig_vibration
  current_channels:
    - motor_current
  scalar_channels:
    - voltage
    - water_temperature
    - pwm_command
```

This profile produces the 2109-dimensional input vector:

```text
2 vibration channels x 1000 samples = 2000
1 current channel x 100 decimated samples = 100
3 scalar channels x 3 statistics = 9
total = 2109
```

Baselines are configured under `baselines`. Faults are configured under `fault_profiles`. Trial groups are configured under `trial_sets`.

Example trial set:

```yaml
healthy_training:
  duration_s: 3
  trials:
    - trial_id: 2026-05-14_POC_B0_nominal_T001
      baseline: B0_nominal
```

For a fault trial, include a fault label and fault timing:

```yaml
known_fault_dictionary:
  duration_s: 3
  fault_start_s: 1.0
  fault_end_s: 2.5
  trials:
    - trial_id: 2026-05-14_POC_B0_fault_bearing_T001
      baseline: B0_nominal
      fault: bearing_impulse
```

## 4. Generate Raw Synthetic Trials

Generate raw trial folders:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli attach-data `
  --source synthetic `
  --config configs/poc_synthetic_training_smoke.yaml `
  --out data/raw/trials_training_smoke
```

Each generated trial folder contains:

```text
manifest.yaml
telemetry.parquet
events.csv
notes.md
quality_report.json
```

The actual sensor data is in `telemetry.parquet`. The `events.csv` file only contains event markers such as trial start, fault start, fault end, and trial end.

Raw trial folders are treated as immutable. If the canonical files already exist, the synthetic attachment command reuses the trial and reruns QC instead of overwriting it.

## 5. Quality Check a Trial

Run QC on a generated trial:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli qc `
  --trial data/raw/trials_training_smoke/2026-05-14_POC_B0_nominal_T001
```

QC checks:

- required files exist
- manifest can be parsed
- telemetry channels exist
- timestamps are monotonic
- sample rate is close to the expected 10 kHz
- channels are not all null
- fault trials contain fault start and fault end events

The QC result is written to:

```text
quality_report.json
```

## 6. Create a Telemetry Preview

Create a small CSV preview and channel summary:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli preview `
  --trial data/raw/trials_training_smoke/2026-05-14_POC_B0_nominal_T001
```

This writes:

```text
telemetry_preview.csv
telemetry_preview_summary.csv
```

Use this to inspect signal ranges quickly without opening the full Parquet file.

## 7. Configure Dataset Generation

Dataset generation is controlled by `dataset_poc_synthetic*.yaml`.

Key fields:

```yaml
dataset_id: ds_poc_synthetic_training_smoke
raw_trial_root: data/raw/trials_training_smoke

windowing:
  window_ms: 100
  stride_ms: 100

preprocessing:
  vibration_sample_rate_hz: 10000
  current_sample_rate_hz: 1000
  scalar_features: [mean, variance, peak_to_peak]
  expected_input_dim: 2109

split:
  strategy: by_trial
  train:
    - 2026-05-14_POC_B0_nominal_T001
  validation:
    - 2026-05-14_POC_B0_nominal_T002
  test:
    - 2026-05-14_POC_B0_fault_bearing_T001
```

The split is by trial, not by random window. This avoids leakage from adjacent windows.

The training split should contain healthy windows only for SDAE training.

## 8. Build the Processed Dataset

Create processed windows:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli make-dataset `
  --config configs/dataset_poc_synthetic_training_smoke.yaml `
  --out data/processed/datasets/ds_poc_synthetic_training_smoke
```

This writes:

```text
dataset_manifest.yaml
split_manifest.yaml
windows.parquet
labels.parquet
```

`windows.parquet` contains numeric model inputs. `labels.parquet` contains metadata for each window:

```text
trial_id
window_start_s
window_end_s
baseline_id
baseline_name
fault_label
is_fault
split
```

For the current report profile, each row in `windows.parquet` should have 2109 features.

## 9. Configure the SDAE

SDAE training is controlled by `baseline_sdae*.yaml`.

Smoke config:

```yaml
model:
  input_dim: 2109
  hidden_dims: [128, 64]
  latent_dim: 16
  hidden_activation: relu
  output_activation: sigmoid
  masking_noise: 0.10
  l1_lambda: 0.0001

training:
  optimizer: adam
  learning_rate: 0.001
  batch_size: 8
  epochs: 8
  early_stopping: false
  seed: 20260514

threshold:
  method: validation_percentile
  target_false_positive_rate: 0.02
```

Full thesis-default config:

```yaml
model:
  input_dim: 2109
  hidden_dims: [2048, 1024]
  latent_dim: 420
  hidden_activation: relu
  output_activation: sigmoid
  masking_noise: 0.30
  l1_lambda: 0.0001
```

The current architecture is:

```text
input -> Linear/ReLU hidden encoder layers -> linear latent
latent -> Linear/ReLU hidden decoder layers -> Linear/Sigmoid output
```

The scaler is fitted only on healthy training windows. Validation windows are used to select the reconstruction threshold.

## 10. Train the SDAE

Train the smoke SDAE:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli train-sdae `
  --dataset data/processed/datasets/ds_poc_synthetic_training_smoke `
  --config configs/baseline_sdae_smoke.yaml `
  --out artifacts/models/run_poc_sdae_smoke
```

The command writes:

```text
run_manifest.yaml
config.yaml
model.pt
scaler.joblib
threshold.json
training_history.csv
metrics.json
plots/loss_curve.png
plots/reconstruction_error_hist.png
```

Despite the `.joblib` name, the scaler is currently a standard Python pickle file because the local environment does not have `joblib` installed. The filename is kept to match the planned artifact naming.

## 11. Inspect the Training Result

Start with:

```text
artifacts/models/run_poc_sdae_smoke/metrics.json
```

Important fields:

```text
train_reconstruction_error_mean
validation_reconstruction_error_mean
healthy_false_positive_rate
true_fault_detection_rate
fault_reconstruction_error_mean
fault_error_greater_than_train_error
loss_decreased
```

Then inspect:

```text
threshold.json
training_history.csv
plots/loss_curve.png
plots/reconstruction_error_hist.png
```

The plots are simple PNG artifacts generated without Matplotlib. They are useful for checking the loss curve and validation reconstruction-error distribution. They are not architecture diagrams or latent-space visualisations.

## 12. One-Command Verification

To run the current end-to-end smoke training workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_objective_3_checks.ps1
```

Expected successful output should include:

```text
Attached 3 synthetic trials under data\raw\trials_training_smoke
Created dataset ds_poc_synthetic_training_smoke with 90 windows, 2109 features, and 3 trials
Trained run_poc_sdae_smoke for 8 epochs on 30 healthy windows
Ran 5 tests
OK
```

## 13. Common Edits

To train longer, edit:

```yaml
training:
  epochs: 50
```

To widen the model, edit:

```yaml
model:
  hidden_dims: [512, 256]
  latent_dim: 64
```

To use the planned full model, use:

```text
configs/baseline_sdae.yaml
```

To add another synthetic healthy trial, add it under a `healthy_training` trial set and include the trial ID in the dataset split.

To add another fault trial, add it under a fault trial set, set `fault_start_s` and `fault_end_s`, and ensure the fault label exists under `fault_profiles`.

## 14. Current Limitations

- Only synthetic data is supported.
- The training command does not yet export latent vectors.
- HDBSCAN and fault dictionary generation are not implemented yet.
- There is no live hardware ingestion yet.
- The plotting helper only creates simple loss/error PNGs.
- The current smoke config is for fast validation, not final thesis evidence.

For final thesis runs, use longer trials, the full model config, and later the fault dictionary/evaluation milestones once implemented.
