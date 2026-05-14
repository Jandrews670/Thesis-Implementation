# Data Collection and Training System Plan

This document defines the practical system for collecting telemetry, storing training data, building datasets, running training jobs, and keeping model results reproducible. The goal is to avoid a common failure mode in experimental ML projects: collecting useful data but losing track of which sensor setup, baseline condition, preprocessing code, model config, and training run produced each result.

The guiding rule is:

```text
raw recordings are immutable; processed datasets and model artifacts are regenerated from raw recordings plus versioned configs.
```

## 1. Overall Data Flow

The system should have a clear chain of custody from the physical test rig to final report metrics:

```text
live test rig
  -> raw trial recording
  -> trial manifest
  -> quality check report
  -> processed window dataset
  -> dataset manifest
  -> model training run
  -> trained model artifact
  -> replay/live evaluation
  -> metrics table and plots
```

Each stage should write metadata about what produced it. This makes it possible to rerun a training job months later and know whether a result changed because the model improved, the preprocessing changed, or the input data changed.

## 2. Storage Layout

Use the following directory layout under `implementation/` once coding begins:

```text
implementation/
  data/
    raw/
      trials/
        2026-05-03_B0_nominal_T001/
          manifest.yaml
          telemetry.bin
          telemetry_preview.csv
          events.csv
          notes.md
          quality_report.json
        2026-05-03_B1_voltage_T002/
          manifest.yaml
          telemetry.bin
          telemetry_preview.csv
          events.csv
          notes.md
          quality_report.json
    processed/
      datasets/
        ds_0001_baseline_sdae/
          dataset_manifest.yaml
          windows.parquet
          labels.parquet
          scaler.joblib
          split_manifest.yaml
        ds_0002_cross_domain/
          dataset_manifest.yaml
          windows.parquet
          labels.parquet
          scaler.joblib
          split_manifest.yaml
    interim/
      decoded_trials/
      calibration_checks/
    external/
      datasheets/
      reference_measurements/
  artifacts/
    models/
      run_0001_sdae_baseline/
        run_manifest.yaml
        config.yaml
        model.pt
        scaler.joblib
        threshold.json
        training_history.csv
        metrics.json
      run_0002_sdae_hdbscan/
      run_0003_fedrep/
      run_0004_dann/
    dictionaries/
      dict_0001_baseline0/
        dictionary.json
        source_model.txt
        source_dataset.txt
    exports/
      pi_runtime/
  runs/
    reports/
    plots/
    logs/
```

Keep `data/raw/trials` as the source of truth. Never edit raw binary files after collection. If a trial has a problem, record that in its manifest or quality report rather than modifying the data.

## 3. Trial Naming and IDs

Each recording session should have a unique trial ID:

```text
YYYY-MM-DD_B<baseline>_<short-condition>_T<trial-number>
```

Examples:

```text
2026-05-03_B0_nominal_T001
2026-05-04_B1_voltage_12v_T006
2026-05-05_B2_biofouling_T011
2026-05-06_B3_thermal_30c_T017
2026-05-07_B4_ventilation_T023
```

This makes files sortable and keeps baseline information visible without opening metadata files.

## 4. Raw Trial Contents

Each trial folder should contain:

- `manifest.yaml`: required metadata for the run.
- `telemetry.bin`: raw binary packets from the Teensy, stored exactly as received.
- `telemetry_preview.csv`: a small decoded preview for quick plotting and sanity checks.
- `events.csv`: manually or automatically recorded event markers.
- `notes.md`: human notes about anything unusual.
- `quality_report.json`: generated after collection to record dropped packets, duration, sample counts, and obvious sensor problems.

The raw binary file should be preferred over CSV for full-rate data because 10 kHz multi-channel telemetry will grow quickly. CSV is useful for previews, but inefficient and easier to corrupt accidentally.

## 5. Trial Manifest Schema

Every trial should have a manifest like this:

```yaml
trial_id: 2026-05-03_B0_nominal_T001
created_at: "2026-05-03T14:21:00+10:00"
operator: "Jack"
hardware:
  pi_id: "pi5_test_rig"
  teensy_id: "teensy41_01"
  motor: "AspiQueen U01"
  esc: "integrated"
  power_supply: "bench_supply_01"
sensor_config:
  sample_rate_hz: 10000
  accelerometer_channels: ["motor_ax", "motor_ay", "motor_az", "rig_ax", "rig_ay", "rig_az"]
  current_channels: ["motor_current"]
  scalar_channels: ["voltage", "water_temperature", "pwm_command"]
  adc_resolution_bits: 12
baseline:
  id: 0
  name: "nominal"
  voltage_v: 16.0
  water_temperature_c: 22.0
  biofouling: false
  shaker_motor: false
  ventilation_profile: false
fault:
  induced: false
  type: "none"
  start_time_s: null
  end_time_s: null
collection:
  duration_s: 300
  packet_format_version: 1
  firmware_version: "0.1.0"
  software_version: "0.1.0"
notes: "Healthy nominal baseline collection."
```

The exact fields can evolve, but the manifest should always capture enough context to explain the data without relying on memory.

## 6. Event Logging

The `events.csv` file should record time-aligned events:

```csv
timestamp_s,event_type,value,notes
0.000,trial_start,,Motor at nominal PWM
60.000,pwm_command,0.50,Steady state
120.000,fault_start,bearing_drag,Manual fault induced
180.000,fault_end,bearing_drag,Fault removed
300.000,trial_end,,
```

Use this for induced faults, voltage changes, temperature targets, PWM changes, shaker motor changes, and any visible anomaly during collection. These labels are not used to train the unsupervised model directly, but they are essential for evaluation.

## 7. Collection Workflow

The collection process should be boring and repeatable:

1. Create a trial folder and draft `manifest.yaml`.
2. Run a short dry run and verify packet decoding, sensor ranges, and preview plots.
3. Start the actual recording command.
4. Record events during the trial.
5. Stop the recording and immediately generate a quality report.
6. Review preview plots before changing the physical setup.
7. Mark the trial as `accepted`, `accepted_with_notes`, or `rejected` in the manifest.

Example command shape:

```powershell
usv-faults collect --config configs/acquisition.yaml --trial data/raw/trials/2026-05-03_B0_nominal_T001
usv-faults qc --trial data/raw/trials/2026-05-03_B0_nominal_T001
usv-faults preview --trial data/raw/trials/2026-05-03_B0_nominal_T001
```

The first live collector should do less, not more. It only needs to reliably capture packets and metadata. Sophisticated live plots can wait.

## 8. Minimum Dataset Collection Plan

The initial report defines five environmental baselines. For each baseline, collect healthy data first, then fault data.

Suggested minimum:

- Baseline 0 nominal: longest and cleanest healthy dataset, used for initial SDAE training.
- Baseline 1 power cycling: healthy data at 16 V and 12 V, plus transition periods if useful.
- Baseline 2 bio-fouling: healthy data with propeller modification.
- Baseline 3 thermal shift: healthy data near 30 C after the water temperature has stabilised.
- Baseline 4 shock/ventilation: healthy data under shaker/PWM disturbance without an actual mechanical fault.
- Fault trials: repeat each induced fault type under Baseline 0 first, then at least selected repeats under Baselines 1-4 for cross-domain evaluation.

Do not start by collecting every possible combination. Start with enough data to validate the pipeline:

- 2-3 healthy nominal trials.
- 1 healthy trial for each shifted baseline.
- 1-2 obvious induced fault trials under Baseline 0.
- 1 induced fault repeat under one shifted baseline.

Once the pipeline works, expand the experimental matrix.

## 9. Data Quality Checks

Every raw trial should pass automated checks before it is used for training:

- Packet loss rate and sequence gaps.
- Actual sample rate versus expected 10 kHz.
- Trial duration and number of samples.
- Sensor saturation or clipping.
- Flatlined channels.
- Timestamp monotonicity.
- Unexpected voltage or temperature range.
- Missing event labels for fault trials.
- Preview plots for current, vibration, voltage, temperature, and PWM command.

The quality report should not decide scientific validity by itself. It should make bad data obvious before it contaminates training.

## 10. Processed Dataset Generation

Processed datasets should be generated from accepted raw trials using a config file. A dataset is not just `windows.parquet`; it is the combination of:

- Raw trial IDs included.
- Preprocessing code version.
- Window size and stride.
- Feature scaling method.
- Decimation settings.
- Train/validation/test split.
- Label/event alignment method.
- Scaler artifact.
- Generated windows and labels.

Example command shape:

```powershell
usv-faults make-dataset --config configs/dataset_baseline_sdae.yaml --out data/processed/datasets/ds_0001_baseline_sdae
```

The dataset manifest should include:

```yaml
dataset_id: ds_0001_baseline_sdae
created_at: "2026-05-08T10:12:00+10:00"
source_trials:
  - 2026-05-03_B0_nominal_T001
  - 2026-05-03_B0_nominal_T002
windowing:
  window_ms: 100
  stride_ms: 100
preprocessing:
  vibration_sample_rate_hz: 10000
  current_sample_rate_hz: 1000
  scalar_features: ["mean", "variance", "peak_to_peak"]
  expected_input_dim: 2109
scaling:
  method: "standard"
  fit_on: "healthy_train_only"
split:
  strategy: "by_trial"
  train: ["2026-05-03_B0_nominal_T001"]
  validation: ["2026-05-03_B0_nominal_T002"]
  test: []
code:
  git_commit: "record when repo is initialized"
  preprocessing_version: "0.1.0"
status: "active"
```

## 11. Split Strategy

Splits should be made by trial, not by randomly mixing windows from the same trial. Random window-level splitting risks leakage because adjacent windows are highly correlated.

Use these split rules:

- Train the baseline SDAE on healthy windows only.
- Fit scalers on healthy training windows only.
- Use held-out healthy windows to set the reconstruction threshold.
- Use fault trials only for threshold validation and final metric reporting, not for fitting the SDAE.
- For cross-domain testing, train or build the fault dictionary on Baseline 0 first, then evaluate on Baselines 1-4.
- Keep final test trials untouched until the pipeline is mostly stable.

If data is scarce early on, use a temporary development split, but mark it clearly as `dev_only` in the dataset manifest.

## 12. Training Run Structure

Each training run should write a self-contained artifact folder:

```text
artifacts/models/run_0001_sdae_baseline/
  run_manifest.yaml
  config.yaml
  model.pt
  scaler.joblib
  threshold.json
  training_history.csv
  metrics.json
  plots/
    loss_curve.png
    reconstruction_error_hist.png
    latent_projection.png
```

The run manifest should include:

```yaml
run_id: run_0001_sdae_baseline
created_at: "2026-05-08T15:34:00+10:00"
run_type: "baseline_sdae"
dataset_id: ds_0001_baseline_sdae
config_file: configs/baseline_sdae.yaml
model:
  input_dim: 2109
  hidden_dims: [2048, 1024]
  latent_dim: 420
  masking_noise: 0.30
  l1_lambda: 0.0001
training:
  optimizer: "adam"
  learning_rate: 0.001
  batch_size: 256
  epochs: 100
  early_stopping: true
threshold:
  method: "validation_percentile"
  target_false_positive_rate: 0.02
artifacts:
  model: model.pt
  scaler: scaler.joblib
  threshold: threshold.json
```

The model file alone is not enough. A model is only meaningful with its scaler, preprocessing config, threshold, and dataset manifest.

## 13. Training Workflow

The training workflow should be standardised:

1. Select or generate a processed dataset.
2. Load the training config.
3. Train the model.
4. Save the model, scaler reference, training history, and config.
5. Compute validation reconstruction errors.
6. Select the reconstruction threshold.
7. Run replay evaluation on held-out healthy and fault trials.
8. Save metrics and plots.
9. Mark the run status as `candidate`, `rejected`, or `accepted`.

Example command shape:

```powershell
usv-faults train-sdae --dataset data/processed/datasets/ds_0001_baseline_sdae --config configs/baseline_sdae.yaml --out artifacts/models/run_0001_sdae_baseline
usv-faults evaluate --model artifacts/models/run_0001_sdae_baseline --dataset data/processed/datasets/ds_0001_baseline_sdae
```

Training should be able to run without the test rig connected. Live hardware should not be part of the critical path for model iteration once recordings exist.

## 14. Threshold Selection

The reconstruction threshold is a first-class artifact, not a magic number in code.

Start with a validation-percentile method:

- Train on healthy training windows.
- Run inference on healthy validation windows.
- Choose a threshold that targets false positive rate <= 2 percent.
- Save the threshold and the distribution used to calculate it.
- Then test against induced fault trials to measure true fault detection rate.

Later, compare this with alternatives such as mean plus `k` standard deviations or baseline-specific thresholds. The final choice should be based on the report metrics, not convenience.

## 15. HDBSCAN and Fault Dictionary Training

The fault dictionary should be built as a separate artifact from the SDAE model.

Workflow:

1. Load an accepted SDAE model run.
2. Replay Baseline 0 fault trials through the model.
3. Store latent vectors for fault-triggered windows.
4. Run HDBSCAN over the rolling 300-window buffers.
5. Build dictionary entries for stable non-healthy clusters.
6. Estimate covariance with Ledoit-Wolf shrinkage.
7. Save dictionary entries with source trial IDs and source model ID.

Dictionary artifact:

```text
artifacts/dictionaries/dict_0001_baseline0/
  dictionary.json
  dictionary_manifest.yaml
  cluster_summary.csv
  cluster_plots/
```

The dictionary manifest should explicitly state which model generated its latent space. If the encoder changes, the dictionary should be considered stale unless it has been regenerated.

## 16. FedRep Data and Training Management

For FedRep, treat each environmental baseline as a simulated client at first:

```text
client_B0 -> nominal healthy/fault data
client_B1 -> voltage-shift data
client_B2 -> biofouling data
client_B3 -> thermal-shift data
client_B4 -> shock/ventilation data
```

Each client should have:

- Its own local healthy dataset.
- Its own local decoder state.
- A shared/global encoder state.
- A client manifest listing source trials and baseline conditions.

FedRep run artifacts should store:

- Initial encoder.
- Per-round averaged encoder.
- Per-client decoder checkpoints.
- Per-round metrics.
- Fault dictionary regenerated after encoder updates.

This allows the final report to show whether cross-domain performance improved because of representation sharing, not because the evaluation data changed.

## 17. DANN Data and Training Management

DANN requires domain labels for healthy baseline data. The labels can come directly from trial manifests:

```text
B0 = nominal
B1 = power cycling
B2 = biofouling
B3 = thermal shift
B4 = shock/ventilation
```

DANN dataset generation should therefore create both:

- Reconstruction target: the original clean input window.
- Domain target: the baseline ID from the source trial.

DANN training artifacts should store:

- SDAE pretraining run ID.
- DANN feature extractor checkpoint.
- Decoder checkpoint.
- Domain classifier checkpoint for audit, even if not deployed.
- Lambda schedule.
- Domain classifier loss/accuracy history.
- Final deployed model without the classifier.

The domain classifier is not part of the edge diagnostic system after training, but its performance is part of the evidence that the latent space became domain-invariant.

## 18. Model Registry Rules

Keep a simple model registry as a CSV or YAML file:

```text
artifacts/model_registry.csv
```

Suggested columns:

```csv
run_id,run_type,dataset_id,status,created_at,model_path,threshold_path,dictionary_id,notes
run_0001_sdae_baseline,baseline_sdae,ds_0001_baseline_sdae,accepted,2026-05-08,artifacts/models/run_0001_sdae_baseline/model.pt,,,"Initial reconstruction model"
```

Only `accepted` runs should be used for final experiments or Raspberry Pi deployment. Development runs can remain on disk, but should not be confused with thesis results.

## 19. Backup and Versioning

Raw data will be difficult to recreate, so it needs a conservative backup policy:

- Back up `data/raw/trials` after each collection day.
- Keep at least one backup separate from the Raspberry Pi.
- Do not rely on Git for large data files.
- Put code, configs, manifests, and small metrics files in Git.
- Use checksums for raw telemetry files and record them in manifests.
- If storage becomes large, archive rejected trials separately rather than deleting them immediately.

If a data versioning tool is needed later, use DVC. Do not introduce it on day one unless normal folders become unmanageable.

## 20. Minimal First Build

The first useful version of this system should implement only:

- Raw trial folder creation.
- Manifest writing.
- Binary telemetry capture.
- Quality report generation.
- Dataset generation from accepted trials.
- Baseline SDAE training run artifact folder.
- Reconstruction threshold artifact.
- Replay evaluation.

This is enough to support the first real thesis milestone: proving that healthy baseline data can train an SDAE and that induced faults cause measurable reconstruction error.

Everything else, including FedRep, DANN, model export, and richer experiment tracking, should build on that same data system rather than inventing a second workflow later.
