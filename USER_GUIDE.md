# User Guide: Configure and Train a Proof-of-Concept SDAE Model

This guide explains how to configure synthetic proof-of-concept data, generate processed windows, and train a Sparse Denoising Autoencoder (SDAE) model using the current implementation.

The current implementation supports:

- synthetic telemetry generation
- raw trial quality checks
- telemetry preview CSVs
- 100 ms window generation
- SDAE training on healthy windows
- reconstruction-threshold selection
- HDBSCAN latent clustering
- Ledoit-Wolf/Mahalanobis fault dictionary generation
- synthetic POC evaluation reports
- raw-trial replay decision logs
- public CWRU bearing dataset attachment as a reduced vibration-only profile
- public IMS, FEMTO/PRONOSTIA, HUST, and Paderborn bearing dataset attachment templates
- training artifacts and basic plots

It does not yet support production FedRep/DANN validation or live Teensy/Raspberry Pi collection. Those remain later milestones and require real domain/hardware data.

## 1. Environment Setup

Run all commands from:

```powershell
C:\Users\jacks\OneDrive\Thesis\Implementation
```

Create or refresh the local environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

The project currently runs through the local `.venv` and `PYTHONPATH=src`. The package dependencies in `pyproject.toml` include the ML libraries used by the thesis pipeline: PyTorch, HDBSCAN, scikit-learn, SciPy, joblib, and Matplotlib.

On this Windows/Python 3.9 environment, `hdbscan==0.8.40` is pinned because it has a prebuilt wheel. Newer HDBSCAN releases may require Microsoft C++ Build Tools to compile from source.

For manual commands in PowerShell, set:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "src"
```

Then call the CLI as:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli --help
```

## 1.1 Containerised Linux Setup

Use Docker when you want the Windows development machine and Raspberry Pi to run the same Linux software environment. The project image is a full training/evaluation image, not an inference-only image. It installs PyTorch, HDBSCAN, SciPy, scikit-learn, Matplotlib, OpenBLAS/LAPACK, and compiler tools so the Pi can run training and dictionary generation as well as replay/evaluation.

For the detailed Raspberry Pi checklist, including Docker Engine installation, serial-device mounting, and cleanup commands, see:

```text
RASPBERRY_PI_SETUP.md
```

From Windows, start Docker Desktop first, then run:

```powershell
.\scripts\docker_build.ps1
.\scripts\docker_test.ps1
```

By default the Docker build installs CPU-only PyTorch from:

```text
https://download.pytorch.org/whl/cpu
```

This keeps the image much smaller than the default Linux PyPI Torch install, which may download CUDA packages that are not useful on Raspberry Pi. If the Pi/ARM64 build cannot resolve a CPU-index Torch wheel, fall back to normal PyPI resolution:

```powershell
.\scripts\docker_build.ps1 -TorchIndexUrl ""
```

Open a Linux shell in the project container:

```powershell
.\scripts\docker_shell.ps1
```

Build an ARM64 image from Windows for Raspberry Pi compatibility testing:

```powershell
.\scripts\docker_build.ps1 -Platform linux/arm64 -Tag usv-faults:pi
```

From Raspberry Pi OS or another 64-bit Linux install on the Pi:

```bash
uname -m
getconf LONG_BIT
dpkg --print-architecture
bash scripts/docker_build.sh
bash scripts/docker_test.sh
```

The expected architecture checks are `aarch64`, `64`, and `arm64`. If the Pi reports a 32-bit userland, reinstall a 64-bit OS before trying to run the full training container.

Linux fallback to normal PyPI Torch resolution:

```bash
TORCH_INDEX_URL= bash scripts/docker_build.sh
```

The container smoke test runs:

```text
python -m unittest discover -s tests
python -m usv_faults.cli --help
Objective 1 smoke path
Objective 2 smoke path
Objective 3 smoke path
Objective 4 smoke path
Objective 5 smoke path
```

You can run individual Linux smoke scripts inside or outside Docker:

```bash
bash scripts/run_objective_1_checks.sh
bash scripts/run_objective_2_checks.sh
bash scripts/run_objective_3_checks.sh
bash scripts/run_objective_4_checks.sh
bash scripts/run_objective_5_checks.sh
```

The container mounts the repository into `/app`, so generated `data/`, `artifacts/`, and `runs/` folders appear on the host machine. These folders are ignored during image builds so trained models and datasets are not baked into the Docker image.

For future live Teensy/Raspberry Pi serial work, pass the serial device into the container:

```bash
docker compose run --rm --device /dev/ttyACM0:/dev/ttyACM0 usv-faults bash
```

If the Pi user cannot access the device, add the user to the relevant Linux group, commonly `dialout`, then log out and back in:

```bash
sudo usermod -aG dialout "$USER"
```

Containerisation standardises Python and Linux dependencies, but target-hardware measurements still need to be collected on the Pi. In particular, CPU usage, RAM usage, power draw, serial latency, and thermal throttling cannot be proven by the Windows container alone.

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

Use the public CWRU configs when you want to run the implemented Objective 7 public-data check through the current Objective 2-5 pipeline:

```text
configs/public_cwru.yaml
configs/dataset_public_cwru.yaml
configs/baseline_sdae_public_cwru.yaml
configs/hdbscan_public_cwru.yaml
```

This is a reduced vibration-only public bearing dataset path. It is not padded to the 2109-dimensional USV schema.

Use these additional public dataset config sets after downloading and extracting the source files locally:

```text
configs/public_ims.yaml
configs/dataset_public_ims.yaml
configs/baseline_sdae_public_ims.yaml
configs/hdbscan_public_ims.yaml

configs/public_femto.yaml
configs/dataset_public_femto.yaml
configs/baseline_sdae_public_femto.yaml
configs/hdbscan_public_femto.yaml

configs/public_hust.yaml
configs/dataset_public_hust.yaml
configs/baseline_sdae_public_hust.yaml
configs/hdbscan_public_hust.yaml

configs/public_paderborn.yaml
configs/dataset_public_paderborn.yaml
configs/baseline_sdae_public_paderborn.yaml
configs/hdbscan_public_paderborn.yaml
```

These are local-file adapters, not guaranteed one-command downloaders. Public archives often unpack with different folder names, so check the `path`, `records`, `columns`, and `mat_variables` fields before running them.

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

The scaler artifact keeps the planned filename `scaler.joblib`. The current scaler class is still project-local, but the broader Milestone 4 environment now includes the real `joblib`, scikit-learn, SciPy, HDBSCAN, and Matplotlib dependencies.

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

- Public CWRU has the automated Objective 7 check. IMS, FEMTO, HUST, and Paderborn have tested local-file adapters and config templates, but they still require manual dataset download/extraction and path verification.
- Latent vectors are currently exported by `build-dictionary`, not by `train-sdae`.
- HDBSCAN, Ledoit-Wolf covariance, and chi-square thresholds require the package dependencies listed in `pyproject.toml`.
- There is no live hardware ingestion yet; `run` currently supports replay only.
- CPU and RAM are measured for training/evaluation reports, but power and thermal behaviour still require target Raspberry Pi hardware measurements.
- The current smoke config is for fast validation, not final thesis evidence.

For final thesis runs, use longer trials and the full model config before interpreting metrics scientifically.

## 15. Build the Fault Dictionary

Dictionary generation is controlled by `configs/hdbscan.yaml`.

Current smoke/default parameters:

```yaml
rolling_window_size: 30
min_cluster_size: 15
min_samples: 15
metric: euclidean
cluster_selection_method: eom
allow_single_cluster: true
mahalanobis_confidence: 0.99
mahalanobis_empirical_enabled: true
mahalanobis_empirical_percentile: 0.95
mahalanobis_empirical_margin: 1.0
mahalanobis_empirical_min_samples: 5
min_runtime_cluster_size: 15
cluster_match_min_member_fraction: 0.50
dictionary_baseline_id: 0
known_fault_labels:
  - bearing_impulse
  - propeller_imbalance
withheld_fault_labels:
  - shaft_rub
```

The dictionary builder:

- extracts reconstruction errors and latent vectors from the trained SDAE
- selects anomaly windows using the saved SDAE threshold
- builds dictionary entries only from Baseline 0 known fault anomaly windows
- excludes labels listed in `withheld_fault_labels`
- clusters candidate latents using `hdbscan.HDBSCAN`
- estimates each cluster covariance and precision matrix using `sklearn.covariance.LedoitWolf`
- computes the theoretical squared-Mahalanobis known/novel threshold with `scipy.stats.chi2.ppf`
- optionally tightens each entry with a source-cluster empirical Mahalanobis radius

With `mahalanobis_empirical_enabled: true`, the stored `mahalanobis_threshold` is the effective threshold used at runtime. The original chi-square boundary is still recorded as `mahalanobis_chi_square_threshold`. Set `mahalanobis_empirical_enabled: false` to use the original chi-square-only gate.

At evaluation/replay time, known/novel decisions are made from a rolling HDBSCAN cluster rather than from one isolated latent point. The runtime path keeps the last 30 latent windows, clusters the anomalous latents inside that temporal buffer, finds the current point's runtime cluster, compares that cluster centroid to stored dictionary centroids, and also requires a configurable fraction of runtime cluster members to fall inside the stored Mahalanobis boundary.

For the current smoke model, `latent_dim: 16`, so the 99 percent Mahalanobis threshold is:

```text
chi2.ppf(0.99, 16) = 31.999926908815176
```

For the thesis-default model, `latent_dim: 420`, so the same rule gives the planned threshold near:

```text
chi2.ppf(0.99, 420) ~= 487.6
```

Build the smoke dictionary from the trained SDAE:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli build-dictionary `
  --model artifacts/models/run_poc_sdae_smoke `
  --dataset data/processed/datasets/ds_poc_synthetic_training_smoke `
  --config configs/hdbscan.yaml `
  --out artifacts/dictionaries/dict_poc_b0_smoke
```

This writes:

```text
dictionary_manifest.yaml
dictionary.json
cluster_summary.csv
latent_windows.parquet
cluster_assignments.csv
known_novel_decisions.csv
cluster_plots/
```

The current `configs/hdbscan.yaml` uses real `hdbscan.HDBSCAN`, `sklearn.covariance.LedoitWolf`, and `scipy.stats.chi2.ppf`. The `shaft_rub` label is configured as withheld from dictionary construction for later known/novel testing.

One-command Objective 4 smoke check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_objective_4_checks.ps1
```

## 16. Objective 4 Test Coverage

`tests/test_objective_4.py` contains two Milestone 4 checks.

`test_ledoit_wolf_and_chi_square_outputs` verifies:

- `covariance_with_ledoit_wolf(...)` uses `sklearn.covariance.LedoitWolf`
- covariance and precision matrices have the expected shape
- Ledoit-Wolf shrinkage is between `0.0` and `1.0`
- `chi_square_threshold(420, 0.99)` uses `scipy.stats.chi2.ppf`
- the 420-dimensional 99 percent threshold is above `480`, matching the thesis expectation near `487.6`

`test_build_dictionary_writes_hdbscan_artifacts` runs a reduced integration path:

```text
synthetic trials -> dataset -> tiny SDAE -> HDBSCAN dictionary
```

It verifies:

- required dictionary files are written
- `dictionary_manifest.yaml` records `hdbscan.HDBSCAN`
- `dictionary_manifest.yaml` records `scipy.stats.chi2.ppf`
- `dictionary.json` records `sklearn.covariance.LedoitWolf`
- at least one dictionary entry is created for `bearing_impulse`

## 17. Evaluate the POC Pipeline

Run evaluation against a processed dataset:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli evaluate `
  --model artifacts/models/run_poc_sdae_smoke_objective_5 `
  --dictionary artifacts/dictionaries/dict_poc_b0_smoke_objective_5 `
  --dataset data/processed/datasets/ds_poc_synthetic_training_smoke `
  --out runs/reports/objective_5_smoke
```

This writes:

```text
poc_detection_metrics.csv
poc_isolation_metrics.csv
poc_event_metrics.csv
poc_event_decisions.csv
poc_cross_domain_metrics.csv
poc_performance_metrics.csv
poc_window_decisions.csv
poc_summary.md
```

Current metric behavior:

- detection, isolation, event, and cross-domain summary metrics exclude the first 10 windows of each contiguous trial/baseline/fault state by default
- raw decision files still include every window and mark skipped rows with `state_window_index`, `metric_excluded`, and `metric_exclusion_reason`
- pass `--metric-warmup-windows 0` to `evaluate` to reproduce the older no-skip metric behavior
- false positive rate is measured on healthy windows
- true fault detection rate is measured on fault windows flagged by the SDAE threshold
- true fault isolation rate is measured on known fault anomaly windows whose rolling runtime cluster matches the correct dictionary label
- event-level false positive, detection, known-fault isolation, withheld-novel, and latency metrics are measured from a rolling vote over recent per-window decisions
- fault isolation latency is measured from the first labelled fault window to the first correct known/novel decision
- DBCV is calculated when the dictionary clustering artifact contains at least two non-noise clusters; the smoke dictionary has one cluster, so DBCV is marked `not_available_single_cluster`
- B1-B4 cross-domain rows are written and marked `not_available` when the dataset has no shifted-baseline known fault anomaly windows
- model artifact size is recorded from the model artifact directory and `model.pt`
- CPU and RAM are measured for training and offline inference benchmarks using process CPU time and resident memory
- FLOP counts are estimated from the SDAE Linear layer dimensions
- power is explicitly not measured in this offline POC command

`poc_performance_metrics.csv` contains:

- `model_parameter_count`
- `model_nonzero_parameter_count`
- `estimated_parameter_memory_fp32_mb`
- `estimated_forward_linear_macs_per_window`
- `estimated_forward_linear_flops_per_window`
- `estimated_training_linear_flops_per_window`
- `estimated_training_linear_flops_total`
- `training_cpu_usage_percent_all_cores`
- `training_peak_ram_mb`
- `inference_cpu_usage_percent_all_cores`
- `inference_peak_ram_mb`
- `inference_wall_time_ms_per_window`
- `inference_throughput_windows_per_second`

The FLOP estimates are linear-layer estimates only. Forward FLOPs count multiply-adds as 2 FLOPs plus bias additions. Training FLOPs are approximated as 3x forward FLOPs per training window. Activations, optimizer bookkeeping, data loading, HDBSCAN, Pandas/PyArrow work, and Python overhead are not included in the static FLOP estimate, but CPU/RAM measurements do include real process overhead during the measured blocks.

The event layer does not replace `poc_window_decisions.csv`. It writes `poc_event_decisions.csv` by counting recent anomaly, known-label, and novel votes over a rolling event window. Sustained anomalies with enough matching known votes become an event-level `known` fault; sustained anomalies without a known majority become event-level `novel`.

The latest Mahalanobis confidence sweeps are generated under `runs/reports/mahalanobis_confidence_sweep/`. They compare chi-square-only matching with empirical `p=0.74` matching across CWRU, expanded IMS, and FEMTO while excluding the first 10 windows of each state from metrics. The empirical `p=0.74` gate is the current best-supported setting for the public-data evidence because it fixes the IMS withheld-novel roller-fault case that chi-square-only matching fails.

The smoke Objective 5 run currently reports:

```text
false_positive_rate: 0.04
true_fault_detection_rate: 1.0
true_fault_isolation_rate: 1.0
cross-domain B1-B4 status: not_available
```

## 18. Replay a Raw Trial

Replay a raw synthetic trial through the trained model and dictionary:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli run `
  --source replay `
  --trial data/raw/trials_training_smoke/2026-05-14_POC_B0_fault_bearing_T001 `
  --model artifacts/models/run_poc_sdae_smoke_objective_5 `
  --dictionary artifacts/dictionaries/dict_poc_b0_smoke_objective_5 `
  --out runs/logs/objective_5_smoke
```

The replay command rebuilds 100 ms windows from the raw trial folder, applies the saved scaler and SDAE, maintains a 30-window rolling latent buffer, runs HDBSCAN on the anomalous latents inside that buffer when enough anomaly vectors are available, and applies the Mahalanobis dictionary decision to the current runtime cluster.

Replay writes:

```text
runs/logs/objective_5_smoke/<trial_id>_replay_decisions.csv
```

Columns:

```text
timestamp_s
trial_id
reconstruction_error
threshold
is_anomaly
cluster_label
dictionary_decision
decision_basis
matched_fault_id
matched_fault_label
mahalanobis_distance_sq
mahalanobis_threshold
cluster_support_count
cluster_member_inlier_fraction
```

For the current smoke replay, the B0 `bearing_impulse` trial produced 30 decision rows and 17 anomaly windows. Only windows with enough accumulated anomaly-cluster support can become known; earlier anomaly windows are marked as insufficient support or cluster noise.

One-command Objective 5 smoke check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_objective_5_checks.ps1
```

## 19. Objective 5 Test Coverage

`tests/test_objective_5.py` runs a reduced end-to-end Milestone 5 path:

```text
synthetic trials -> dataset -> tiny SDAE -> HDBSCAN dictionary -> evaluate -> replay
```

It verifies:

- evaluation writes detection, isolation, cross-domain, and summary artifacts
- detection metrics include false positive and true fault detection columns
- isolation metrics include true fault isolation and DBCV status columns
- cross-domain metrics include B1-B4 rows
- performance metrics include FLOP estimates plus training/inference CPU and RAM rows
- replay writes the required decision-log columns
- replay produces at least one anomaly and at least one known dictionary decision

## 20. Objective 7 Public CWRU Check

Objective 7 currently uses the Case Western Reserve University bearing data as the runnable public-data path. Paderborn remains the closer dataset for the final thesis sensor plan because it includes motor current and vibration, but the official Paderborn archives are large RAR files. The CWRU path is therefore used as the lightweight public realism check.

Run the public check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_objective_7_public_checks.ps1
```

This downloads the selected public `.mat` files into:

```text
data/external/cwru/
```

Then it writes:

```text
data/raw/public_cwru/
data/processed/datasets/ds_public_cwru_objective_7/
artifacts/models/run_public_cwru_sdae_objective_7/
artifacts/dictionaries/dict_public_cwru_objective_7/
runs/reports/objective_7_public_cwru/
```

The CWRU profile uses one 12 kHz drive-end vibration channel with 100 ms windows:

```text
1 vibration channel x 1200 samples = 1200 input features
```

No current or scalar channels are fabricated. The raw trial manifests and dataset config record the reduced profile explicitly.

The current public CWRU run produced 300 windows, 8 dictionary entries, a 1.67 percent healthy false positive rate, 100 percent fault-window detection, 83.33 percent known-fault isolation, and 100 percent novel decision rate for the withheld outer-race fault. Treat this as development evidence only; it is not a substitute for the planned USV hardware data.

`tests/test_objective_7.py` keeps CI/offline checks deterministic by creating a tiny local MATLAB fixture, attaching it through the same CWRU adapter, and running dataset generation, SDAE training, dictionary generation, and evaluation against the reduced public-profile contract.

## 21. Additional Public Bearing Adapters

The implementation now includes config-driven local-file adapters for four more public bearing datasets:

```text
IMS/NASA Bearings: --source ims
FEMTO/PRONOSTIA: --source femto
HUST Bearings: --source hust
Paderborn Bearings: --source paderborn
```

Run their fixture checks without downloading the full datasets:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_public_bearing_adapter_checks.ps1
```

After downloading and extracting a dataset, adjust the relevant `configs/public_*.yaml` paths and run the usual pipeline. Example for Paderborn:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli attach-data --source paderborn --config configs/public_paderborn.yaml --out data/raw/public_paderborn
.\.venv\Scripts\python.exe -m usv_faults.cli make-dataset --config configs/dataset_public_paderborn.yaml --out data/processed/datasets/ds_public_paderborn
.\.venv\Scripts\python.exe -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_public_paderborn --config configs/baseline_sdae_public_paderborn.yaml --out artifacts/models/run_public_paderborn_sdae
.\.venv\Scripts\python.exe -m usv_faults.cli build-dictionary --model artifacts/models/run_public_paderborn_sdae --dataset data/processed/datasets/ds_public_paderborn --config configs/hdbscan_public_paderborn.yaml --out artifacts/dictionaries/dict_public_paderborn
.\.venv\Scripts\python.exe -m usv_faults.cli evaluate --model artifacts/models/run_public_paderborn_sdae --dictionary artifacts/dictionaries/dict_public_paderborn --dataset data/processed/datasets/ds_public_paderborn --out runs/reports/public_paderborn
```

The Paderborn template maps one vibration channel and one motor-current channel, giving a 12,800-D 100 ms input at 64 kHz. IMS and HUST use one vibration channel. FEMTO uses horizontal and vertical acceleration channels. None of these adapters pads missing channels to the synthetic 2109-D USV profile.
