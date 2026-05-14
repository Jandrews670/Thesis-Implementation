# Agent Execution Plan for the Proof-of-Concept Implementation

This document is the coding-agent handoff for the first multi-hour implementation sprint. It translates the implementation plan, data collection plan, and proof-of-concept plan into an ordered build sequence with explicit outputs and verification gates.

The goal is not to build the entire thesis system in one pass. The goal is to produce a runnable proof of concept that exercises the same data path the final Raspberry Pi/Teensy system will use:

```text
source adapter
  -> raw trial folder
  -> quality check
  -> processed 100 ms windows
  -> SDAE training
  -> reconstruction threshold
  -> replay anomaly detection
  -> latent clustering
  -> Mahalanobis/Ledoit-Wolf fault dictionary
  -> known/novel decision
  -> evaluation reports
```

Only the source adapter should differ between synthetic data, public datasets, and future hardware recordings.

## 1. Mission, Constraints, and Non-Negotiables

### Mission

Build a Python package and CLI that can run the synthetic proof of concept end to end:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
usv-faults make-dataset --config configs/dataset_poc_synthetic.yaml --out data/processed/datasets/ds_poc_synthetic_0001
usv-faults train-sdae --dataset data/processed/datasets/ds_poc_synthetic_0001 --config configs/baseline_sdae.yaml --out artifacts/models/run_poc_sdae_0001
usv-faults build-dictionary --model artifacts/models/run_poc_sdae_0001 --dataset data/processed/datasets/ds_poc_synthetic_0001 --config configs/hdbscan.yaml --out artifacts/dictionaries/dict_poc_b0_0001
usv-faults evaluate --model artifacts/models/run_poc_sdae_0001 --dictionary artifacts/dictionaries/dict_poc_b0_0001 --dataset data/processed/datasets/ds_poc_synthetic_0001 --out runs/reports
usv-faults run --source replay --trial data/raw/trials/2026-05-14_POC_B0_fault_bearing_T001 --model artifacts/models/run_poc_sdae_0001 --dictionary artifacts/dictionaries/dict_poc_b0_0001
```

### Non-Negotiables

- Use a small Python package, not loose notebooks.
- Keep synthetic data behind a source adapter. Do not let training, clustering, or evaluation code depend on synthetic-only objects.
- Treat raw trial folders as immutable once generated.
- Generate processed datasets from raw trials plus config.
- Fit scalers only on healthy training windows.
- Split by trial, not by random window.
- Keep model dimensions, HDBSCAN settings, thresholds, and dataset definitions in config files.
- Save every model, dictionary, and evaluation run with enough metadata to reproduce it.
- Do not begin FedRep or DANN implementation until the baseline SDAE and dictionary pipeline works.

### Current Status and Watch-Outs

As of 2026-05-14, Milestones 1-3 have been implemented and smoke-verified. Future agents should treat this as a working foundation, not as final experimental evidence.

- Run commands from `Implementation/` using `.\.venv\Scripts\python.exe -m usv_faults.cli` and `PYTHONPATH=src`.
- The environment is Python 3.9 and intentionally avoids mandatory `typer`, `pydantic`, `pytest`, `scikit-learn`, `joblib`, `scipy`, and `matplotlib` dependencies.
- The smoke configs are fast verification configs. The full configs remain the intended POC path and should be run deliberately when runtime/storage are acceptable.
- Raw synthetic signal data is `telemetry.parquet`; `events.csv` is expected to contain only event markers.
- Synthetic `attach-data` should reuse existing canonical raw trial folders rather than overwriting them.
- The SDAE should default to ReLU hidden layers and Sigmoid output activation. Record both as config fields and in `run_manifest.yaml`.
- The current smoke SDAE is deliberately small; do not infer full architecture performance from it.
- Current plots are simple PNG training artifacts, not model architecture visualisations.

### Key Existing Source Documents

- `Implementation/implementation_plan.md`
- `Implementation/data_collection_training_plan.md`
- `Implementation/proof_of_concept_plan.md`
- `Implementation/poc_synthetic_data_attachment.yaml`

## 2. Target Repository Shape

Create the implementation as a package rooted under `Implementation/`:

```text
Implementation/
  pyproject.toml
  README.md
  configs/
    poc_synthetic.yaml
    dataset_poc_synthetic.yaml
    baseline_sdae.yaml
    hdbscan.yaml
  src/
    usv_faults/
      cli.py
      config.py
      schemas.py
      data_sources/
        base.py
        synthetic_usv.py
        replay.py
      storage/
        trials.py
        artifacts.py
      preprocessing/
        windowing.py
        feature_scaling.py
        datasets.py
      models/
        sdae.py
        checkpoints.py
      training/
        train_sdae.py
        threshold_search.py
      clustering/
        hdbscan_pipeline.py
        fault_dictionary.py
        mahalanobis.py
      evaluation/
        metrics.py
        trial_runner.py
        reports.py
      edge/
        realtime_pipeline.py
  tests/
```

Use this structure unless a minor adjustment is required by the packaging tool. Keep module boundaries intact.

## 3. Milestone 0: Package, CLI, and Config Foundation

### Implementation Tasks

- Create `pyproject.toml` with dependencies:
  - runtime: `numpy`, `scipy`, `pandas`, `pyarrow`, `torch`, `scikit-learn`, `hdbscan`, `typer`, `pydantic`, `pyyaml`, `joblib`, `matplotlib`
  - development: `pytest`, `ruff`
- Add a `usv-faults` console entry point targeting `usv_faults.cli:app`.
- Add config loading helpers that read YAML and validate with Pydantic models.
- Add canonical schema models for:
  - trial manifest
  - event row
  - quality report
  - dataset manifest
  - training run manifest
  - dictionary manifest
- Add the initial configs:
  - `configs/poc_synthetic.yaml`, copied from `poc_synthetic_data_attachment.yaml` and adjusted only as needed for implementation.
  - `configs/dataset_poc_synthetic.yaml`
  - `configs/baseline_sdae.yaml`
  - `configs/hdbscan.yaml`

### Required CLI Stubs

Implement these commands early, even if they initially call placeholder functions:

```text
attach-data
qc
preview
make-dataset
train-sdae
build-dictionary
evaluate
run
```

### Verification Gate

Run:

```powershell
pytest
usv-faults --help
usv-faults attach-data --help
```

Success means the package imports, the CLI loads, config parsing works, and the test suite starts cleanly.

## 4. Milestone 1: Synthetic Source Adapter and Raw Trial Folders

### Data Contract

The synthetic adapter must write normal raw trial folders:

```text
data/raw/trials/<trial_id>/
  manifest.yaml
  telemetry.parquet
  events.csv
  notes.md
  quality_report.json
```

Use Parquet for synthetic raw telemetry to simplify early iteration. The future Teensy collector can write `telemetry.bin`; downstream code must support source-specific raw telemetry readers.

### Synthetic Channel Profile

Implement the initial report profile:

```text
raw sample rate: 10 kHz
window size: 100 ms
stride: 100 ms
vibration channels: motor_vibration, rig_vibration
current channel: motor_current
scalar channels: voltage, water_temperature, pwm_command
processed input dimension: 2109
```

The processed dimension is:

```text
2 vibration channels x 1000 samples = 2000
1 current channel x 100 decimated samples = 100
3 scalar channels x 3 stats = 9
total = 2109
```

### Signal Requirements

The synthetic generator must create deterministic but non-trivial telemetry:

- Motor vibration: shaft-frequency fundamental, harmonics, stochastic noise, optional fault modulation.
- Rig vibration: lower-amplitude environmental vibration and low-frequency disturbance.
- Motor current: load-dependent offset, ripple, shaft-frequency relation, fault-dependent modulation.
- Scalars: voltage, water temperature, PWM command.
- Baseline effects:
  - `B0_nominal`: stable 16 V, 22 C, steady PWM.
  - `B1_voltage`: 12 V supply shift.
  - `B2_biofouling`: increased drag and altered resonance.
  - `B3_thermal`: 30 C thermal shift.
  - `B4_shock_ventilation`: low-frequency disturbance plus PWM drop/surge profile.
- Fault effects:
  - `bearing_impulse`
  - `propeller_imbalance`
  - `shaft_rub`
  - `electrical_phase_noise`

Use deterministic random seeds from the config. Record the seed in each manifest.

### Events

Healthy trials must include `trial_start` and `trial_end`.

Fault trials must include:

```csv
timestamp_s,event_type,value,notes
0.000,trial_start,,
60.000,fault_start,<fault_type>,
150.000,fault_end,<fault_type>,
180.000,trial_end,,
```

Use the actual configured `fault_start_s`, `fault_end_s`, and `duration_s`.

### Verification Gate

Run:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
```

Success means all configured synthetic trials are generated, manifests validate, sample rates match, and `quality_report.json` is written.

## 5. Milestone 2: Quality Check, Preview, Windowing, and Dataset Generation

### Quality Checks

Implement QC for synthetic Parquet trials:

- required files exist
- manifest validates
- timestamps are monotonic
- sample rate is within tolerance
- expected channels exist
- no all-null channels
- no flatlined channels
- trial duration approximately matches manifest/config
- fault trials contain fault start and end events

Write `quality_report.json` with:

```text
trial_id
status: accepted | accepted_with_notes | rejected
sample_count
duration_s
sample_rate_hz_estimate
channel_checks
event_checks
warnings
errors
```

### Preview

Implement a minimal `preview` command that writes quick plots or CSV summaries. Do not build a dashboard.

Acceptable first version:

```text
runs/plots/<trial_id>_preview.png
```

with voltage, temperature, PWM, current, and vibration snippets.

### Windowing

Implement deterministic 100 ms non-overlapping windows by default. Keep stride configurable.

For each window:

- flatten vibration at 10 kHz
- decimate current to 1 kHz
- compute scalar mean, variance, and peak-to-peak
- attach labels derived from event overlap:
  - `baseline_id`
  - `baseline_name`
  - `fault_label`
  - `is_fault`
  - `trial_id`
  - `window_start_s`
  - `window_end_s`

### Dataset Outputs

`make-dataset` must write:

```text
data/processed/datasets/ds_poc_synthetic_0001/
  dataset_manifest.yaml
  windows.parquet
  labels.parquet
  split_manifest.yaml
```

Do not save the scaler here unless the dataset command explicitly fits it. For the POC, fit and save the scaler during training to enforce healthy-training-only fitting.

### Verification Gate

Run:

```powershell
usv-faults make-dataset --config configs/dataset_poc_synthetic.yaml --out data/processed/datasets/ds_poc_synthetic_0001
```

Success means:

- `windows.parquet` exists.
- every feature row has exactly 2109 numeric values.
- labels align one-to-one with windows.
- splits are by trial.
- healthy train split contains only B0 healthy trials.

## 6. Milestone 3: Baseline SDAE Training and Reconstruction Detection

### Model

Implement the configurable SDAE:

```text
input_dim: 2109
hidden_dims: [2048, 1024]
latent_dim: 420
decoder: mirror encoder
hidden activation: ReLU
output activation: Sigmoid
masking_noise: 0.30
loss: MSE + L1 latent activation penalty
```

Use PyTorch. Keep the encoder and decoder separable so FedRep and DANN can reuse them later.

The latent projection is currently linear. Keep `hidden_activation` and `output_activation` configurable in YAML so smoke runs can remain small while the full model preserves the intended report architecture.

### Training Rules

- Train only on healthy training windows.
- Fit `StandardScaler` only on healthy training windows.
- Validate on held-out healthy B0 windows.
- Select reconstruction threshold from validation reconstruction errors.
- Default threshold method: validation percentile targeting false positive rate <= 2 percent.
- Save all artifacts under the provided output directory.

### Model Artifact

`train-sdae` must write:

```text
artifacts/models/run_poc_sdae_0001/
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
```

### Verification Gate

Run:

```powershell
usv-faults train-sdae --dataset data/processed/datasets/ds_poc_synthetic_0001 --config configs/baseline_sdae.yaml --out artifacts/models/run_poc_sdae_0001
```

Success means:

- training loss decreases over at least a small smoke-test run
- threshold artifact exists
- healthy validation false positive rate is computed
- B0 fault reconstruction errors are higher than B0 healthy errors in the evaluation metrics

For fast development, allow a config override with fewer epochs and smaller synthetic trial durations, but keep the default config aligned with the POC plan.

## 7. Milestone 4: Latent Clustering and Fault Dictionary

### Latent Extraction

Replay windows through the trained SDAE and store:

```text
trial_id
window_start_s
window_end_s
reconstruction_error
is_anomaly
latent_vector
fault_label
baseline_id
```

### HDBSCAN

Run HDBSCAN on latent vectors from anomaly windows. Start with config values:

```yaml
rolling_window_size: 300
min_cluster_size: 15
min_samples: 15
metric: euclidean
cluster_selection_method: eom
```

Keep all settings configurable.

### Dictionary

Build dictionary entries from stable non-noise clusters:

```text
fault_id
label
centroid
covariance
precision
ledoit_wolf_shrinkage
sample_count
source_trial_ids
source_model_id
source_dataset_id
latent_dim
mahalanobis_confidence
mahalanobis_threshold
```

Use `sklearn.covariance.LedoitWolf` for covariance regularisation.

Compute the Mahalanobis threshold from chi-squared distribution:

```text
threshold = chi2.ppf(confidence, latent_dim)
default confidence = 0.99
```

### Outputs

`build-dictionary` must write:

```text
artifacts/dictionaries/dict_poc_b0_0001/
  dictionary_manifest.yaml
  dictionary.json
  cluster_summary.csv
  cluster_plots/
```

### Verification Gate

Run:

```powershell
usv-faults build-dictionary --model artifacts/models/run_poc_sdae_0001 --dataset data/processed/datasets/ds_poc_synthetic_0001 --config configs/hdbscan.yaml --out artifacts/dictionaries/dict_poc_b0_0001
```

Success means:

- known B0 fault windows form at least one non-noise cluster per intended known fault type
- dictionary entries include Ledoit-Wolf covariance data
- withheld `shaft_rub` is not used to build the dictionary
- dictionary manifest records model and dataset provenance

## 8. Milestone 5: Evaluation, Reports, and Replay Runtime

### Evaluation Metrics

Implement:

- false positive rate
- true fault detection rate
- true fault isolation rate
- fault isolation latency
- DBCV score if available from the HDBSCAN library
- cross-domain accuracy for B1-B4 fault trials
- maximum centroid drift distance
- model artifact size in MB

For POC, CPU/RAM/power metrics may be approximate or deferred with explicit `not_measured` values. Do not fake SWaP-C measurements.

### Reports

`evaluate` must write:

```text
runs/reports/
  poc_detection_metrics.csv
  poc_isolation_metrics.csv
  poc_cross_domain_metrics.csv
  poc_summary.md
```

`poc_summary.md` must clearly state:

- data source is synthetic
- the POC is not final thesis evidence
- model/dictionary/dataset IDs used
- commands needed to reproduce the run
- which acceptance gates passed or failed

### Replay Runtime

Implement `run --source replay` using the same pipeline components:

```text
trial reader -> window builder -> scaler -> model -> reconstruction error -> latent buffer -> clustering/dictionary -> decision log
```

Write:

```text
runs/logs/<trial_id>_replay_decisions.csv
```

with:

```text
timestamp_s
trial_id
reconstruction_error
threshold
is_anomaly
cluster_label
dictionary_decision
matched_fault_id
matched_fault_label
mahalanobis_distance_sq
```

### Verification Gate

Run:

```powershell
usv-faults evaluate --model artifacts/models/run_poc_sdae_0001 --dictionary artifacts/dictionaries/dict_poc_b0_0001 --dataset data/processed/datasets/ds_poc_synthetic_0001 --out runs/reports
usv-faults run --source replay --trial data/raw/trials/2026-05-14_POC_B0_fault_bearing_T001 --model artifacts/models/run_poc_sdae_0001 --dictionary artifacts/dictionaries/dict_poc_b0_0001
```

Success means the reports and replay decision log are produced without touching synthetic-specific code outside the source adapter.

## 9. Milestone 6: FedRep and DANN Dry-Run Scaffolding

Start this only after Milestone 5 passes.

### FedRep Dry Run

Implement only the minimum workflow required to prove artifact and evaluation compatibility:

- Treat B0-B4 as clients.
- Reuse the SDAE encoder and decoder classes.
- Add freeze/unfreeze utilities.
- Train local encoder updates on each client.
- Average encoder weights centrally.
- Rebuild the dictionary after encoder averaging.
- Evaluate B0 dictionary transfer to B1-B4.

Write artifacts under:

```text
artifacts/models/run_poc_fedrep_0001/
```

### DANN Dry Run

Implement only after FedRep scaffolding or if explicitly prioritised later:

- Use healthy B0-B4 windows with domain labels.
- Reuse SDAE encoder.
- Add DANN feature extractor and domain classifier.
- Implement gradient reversal.
- Save the deployed model without the classifier.
- Track domain classifier loss/accuracy for evidence of domain confusion.

Write artifacts under:

```text
artifacts/models/run_poc_dann_0001/
```

### Verification Gate

Success means baseline SDAE, FedRep, and DANN can all be evaluated by the same evaluation command and produce comparable cross-domain metrics.

## 10. Milestone 7: Public Dataset Adapter

Start this after the synthetic POC works.

### Preferred Dataset

Implement a Paderborn adapter first because it includes both current and vibration signals.

The adapter must convert downloaded public data into canonical raw trial folders with:

```yaml
source_type: external_paderborn
source_url: <dataset source>
channel_profile: <explicit reduced or mapped profile>
```

Do not silently fake missing channels to match 2109 dimensions. If the public data profile differs, create a separate dataset config and record the reduced profile in the manifest.

### Verification Gate

Success means the same `make-dataset`, `train-sdae`, `build-dictionary`, and `evaluate` commands run on the external dataset with a clearly labelled POC-only report.

## 11. Test Plan

### Unit Tests

Add tests for:

- config loading and validation
- trial manifest validation
- event parsing and event-to-window label alignment
- deterministic synthetic generation from seed
- QC failure on missing channels and non-monotonic timestamps
- 2109-dimensional feature construction
- scaler fit only on healthy training rows
- SDAE forward pass shape
- threshold percentile calculation
- Mahalanobis distance and chi-squared thresholding
- Ledoit-Wolf dictionary entry creation

### Integration Tests

Add a reduced-duration synthetic config for tests:

```text
2 healthy B0 trials, 1 fault trial, 5-10 seconds each
smaller SDAE dimensions if needed for speed
1-2 epochs
```

The integration test should run:

```text
attach-data -> qc -> make-dataset -> train-sdae -> build-dictionary -> evaluate
```

### Manual Acceptance Run

Before declaring the POC complete, run the full configured POC commands from Section 1 and confirm all expected artifacts exist.

## 12. Stop Conditions and Escalation Points

Stop and report clearly if:

- The 2109-dimensional profile conflicts with the chosen sensor/channel interpretation.
- `hdbscan` installation fails or is not practical on the target machine.
- PyTorch training is too slow for the default SDAE dimensions.
- A future change requires packages not present in the local environment; prefer a self-contained fallback or document the dependency clearly.
- Synthetic faults do not produce separable reconstruction errors after reasonable signal tuning.
- HDBSCAN cannot form stable clusters in the 420-dimensional latent space.
- Public dataset licensing or download structure blocks the adapter.

When blocked, preserve the runnable parts of the pipeline and document the exact command, error, and suggested next decision in `runs/reports/poc_summary.md`.

## 13. Definition of Done

The first implementation sprint is complete when:

- `usv-faults --help` works.
- Synthetic trials can be attached into raw trial folders.
- QC and dataset generation work from raw trial folders.
- Processed windows match the configured input dimension.
- Baseline SDAE trains on healthy B0 synthetic data.
- Reconstruction threshold is saved and used during replay.
- B0 known faults can be clustered and added to a dictionary.
- A withheld synthetic fault is evaluated as novel or explicitly reported as a failed acceptance gate.
- Evaluation reports and replay decision logs are written.
- All artifacts include provenance metadata.
- Tests cover the core data path enough that future hardware adapters can be added without rewriting the pipeline.
