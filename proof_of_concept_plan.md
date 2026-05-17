# Proof of Concept Plan

This proof of concept should de-risk the implementation before the physical test rig and self-collected dataset are ready. It should prove that the planned diagnostic pipeline can ingest trial-like telemetry, build the report's 100 ms input windows, train an SDAE on healthy data, trigger on anomalous reconstruction error, cluster latent vectors with HDBSCAN, and evaluate a fault dictionary. The proof of concept should not become a separate prototype system; it should exercise the same functional code that will later process Teensy/Raspberry Pi recordings.

The key design rule is:

```text
simulation and online datasets enter through source adapters; preprocessing, training, clustering, dictionaries, and evaluation are shared.
```

## 1. Purpose

The proof of concept exists to answer five engineering questions before hardware data collection begins:

- Can the raw trial, manifest, event, windowing, and dataset-generation formats support the thesis pipeline end to end?
- Can the SDAE training code learn healthy electromechanical baselines and produce a usable reconstruction-error threshold?
- Can HDBSCAN over rolling latent vectors produce stable clusters for repeated fault signatures?
- Can the Mahalanobis/Ledoit-Wolf dictionary distinguish known and novel fault clusters?
- Can the FedRep and DANN comparison workflows run against baseline/domain-shifted data without changing the runtime diagnostic pipeline?

The proof of concept is not intended to make final scientific claims about USV performance. Synthetic and public bearing datasets are only development evidence that the pipeline works and that the planned experiments are feasible.

## 2. Shared Code Boundary

The implementation should define a narrow data-source boundary:

```text
RawTrialSource
  -> canonical raw trial folder
  -> make-dataset
  -> train
  -> build-dictionary
  -> evaluate
  -> replay/live run
```

Only `RawTrialSource` changes between proof-of-concept and final hardware work.

Recommended source adapters:

```text
src/usv_faults/data_sources/
  base.py              # RawTrialSource protocol/interface
  synthetic_usv.py     # synthetic USV propulsion telemetry
  paderborn.py         # optional online bearing dataset adapter
  cwru.py              # optional online vibration-only bearing adapter
  serial_teensy.py     # later live/recorded Teensy adapter
```

The shared downstream modules should not know whether the data came from the simulator, a public `.mat` file, or the Teensy. They should only see accepted raw trial folders and manifests matching the data collection plan.

## 3. POC Data Strategy

Use two data sources, in this order:

1. Synthetic USV propulsion telemetry as the primary proof-of-concept source.
2. One public bearing dataset as a secondary realism check.

Synthetic data is preferred first because it can exactly match the intended thesis schema, including voltage, temperature, PWM commands, environmental baselines, event labels, and fault start/end markers. Public datasets are useful because they contain real measured fault signals, but they do not match the planned USV telemetry schema exactly.

### Synthetic Source

The simulator should write normal raw trial folders, not special training arrays:

```text
data/raw/trials/2026-05-14_POC_B0_nominal_T001/
  manifest.yaml
  telemetry.parquet
  events.csv
  notes.md
  quality_report.json
```

For the proof of concept, use the report's 2109-dimensional profile:

```text
2 vibration channels at 10 kHz over 100 ms: 2000 values
1 current channel decimated to 1 kHz over 100 ms: 100 values
3 scalar channels x mean/variance/peak-to-peak: 9 values
total input dimension: 2109
```

Keep the channel profile configurable. The current data collection plan includes a draft six-axis accelerometer manifest; if the final hardware uses six vibration axes, the expected input dimension must be revised from 2109.

Synthetic telemetry should include:

- Motor vibration at a configurable shaft-frequency fundamental plus harmonics.
- Rig/environment vibration with lower-frequency disturbance.
- Motor current with load-dependent offset, ripple, and motor-speed harmonics.
- Voltage, water temperature, and PWM command as scalar channels.
- Sensor noise, small timestamp jitter, and occasional benign impulses.
- Deterministic random seeds recorded in each manifest.

Synthetic baselines should mirror the report:

- `B0_nominal`: 16 V, 22 C, clean propeller, steady PWM.
- `B1_voltage`: voltage shifted toward 12 V.
- `B2_biofouling`: increased drag, altered propeller resonance, higher current draw.
- `B3_thermal`: water temperature shifted toward 30 C with altered current and vibration noise.
- `B4_shock_ventilation`: low-frequency mechanical disturbance plus randomized PWM drops/spikes.

Synthetic faults should be simple but separable:

- `bearing_impulse`: repeated high-frequency impulses and vibration sidebands.
- `propeller_imbalance`: stronger 1x/2x vibration harmonics and slow amplitude modulation.
- `shaft_rub`: intermittent broadband bursts and elevated current draw.
- `electrical_phase_noise`: current ripple and phase-like amplitude modulation without strong vibration change.

Use `shaft_rub` or `electrical_phase_noise` as a withheld novel fault during the first dictionary test.

### Public Dataset Source

Use public data only after the synthetic source validates the full schema.

Best candidate: Paderborn University Bearing DataCenter. It provides synchronously measured motor current and vibration signals, healthy and damaged bearing states, supporting speed/torque/load/temperature measurements, and multiple operating conditions. This is the closest public match to the planned current-plus-vibration thesis telemetry.

Secondary candidate: Case Western Reserve University Bearing Data Center. It provides normal and faulty bearing vibration data from a motor test stand in Matlab format at 12 kHz and 48 kHz, with RPM metadata. It is useful for bearing-fault clustering, but it is less aligned with the planned schema because it is primarily vibration based.

Optional degradation candidate: NASA IMS Bearings. It is useful for testing anomaly onset over bearing degradation, but it is less useful for validating the exact USV multi-sensor schema.

Checked sources, 2026-05-14:

- Paderborn Bearing DataCenter: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
- Paderborn data description/download: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter/welcome
- CWRU data files: https://engineering.case.edu/bearingdatacenter/download-data-file
- NASA IMS Bearings: https://data.nasa.gov/dataset/ims-bearings

## 4. Dataset Matrix

Start small enough to iterate quickly:

| Dataset | Trials | Purpose |
| --- | ---: | --- |
| Synthetic B0 healthy | 3 x 300 s | SDAE training, validation, threshold fitting |
| Synthetic B1-B4 healthy | 1 x 180 s each | false positive and domain-shift checks |
| Synthetic B0 known faults | 2 fault types x 2 trials | initial HDBSCAN and dictionary construction |
| Synthetic B1-B4 known faults | 2 fault types x 1 trial each | cross-domain dictionary evaluation |
| Synthetic withheld novel fault | 1-2 trials | known/novel decision test |
| Paderborn subset | selected healthy/damaged files | real-signal adapter and clustering sanity check |

This matrix is intentionally smaller than the final experiment. Its job is to prove the code path, not to exhaustively tune model performance.

## 5. POC Workflow

### Phase 1: Source Adapter and Trial Attachment

Implement the synthetic adapter first.

Expected command shape:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
usv-faults preview --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
```

The word `attach-data` is deliberate: the simulator attaches data to the same trial storage layout used by real collection. It should not bypass the manifest, event, quality, or raw-data layers.

Deliverable:

- Synthetic trial folders with manifests, telemetry, events, notes, and quality reports.

### Phase 2: Shared Dataset Generation

Run the same dataset builder planned for hardware recordings:

```powershell
usv-faults make-dataset --config configs/dataset_poc_synthetic.yaml --out data/processed/datasets/ds_poc_synthetic_0001
```

The dataset manifest must record:

- Source trial IDs.
- `source_type: synthetic` or `source_type: external_paderborn`.
- Generator or adapter version.
- Random seed for synthetic trials.
- Window size, stride, decimation, scalar feature rules, and expected input dimension.
- Split strategy by trial, not by random windows.

Deliverable:

- Processed windows and labels generated from synthetic raw trials using the shared preprocessing code.

### Phase 3: Baseline SDAE Detection

Train the baseline SDAE only on healthy B0 windows:

```powershell
usv-faults train-sdae --dataset data/processed/datasets/ds_poc_synthetic_0001 --config configs/baseline_sdae.yaml --out artifacts/models/run_poc_sdae_0001
```

Use the planned architecture as the default:

```text
input: 2109
encoder: 2048 -> 1024
latent: 420
decoder: 1024 -> 2048
output: 2109
masking noise: 30 percent
loss: MSE + L1 latent activation penalty
```

Acceptance criteria:

- Healthy validation reconstruction error is stable enough to set a threshold.
- B0 fault windows show higher reconstruction error than B0 healthy windows.
- B1-B4 healthy trials expose whether the threshold is over-sensitive to baseline shifts.

Deliverable:

- SDAE model artifact, scaler, reconstruction threshold, training history, and reconstruction-error plots.

### Phase 4: HDBSCAN and Fault Dictionary

Replay B0 fault trials through the trained SDAE, store latent vectors, and cluster the rolling latent buffer:

```powershell
usv-faults build-dictionary --model artifacts/models/run_poc_sdae_0001 --dataset data/processed/datasets/ds_poc_synthetic_0001 --config configs/hdbscan.yaml --out artifacts/dictionaries/dict_poc_b0_0001
```

Use the report's 300-window rolling buffer as the initial setting. With 100 ms non-overlapping windows, this represents 30 seconds of telemetry.

Dictionary entries should include:

- Fault label if known from events.
- Cluster centroid.
- Ledoit-Wolf covariance/shrinkage data.
- Sample count and source trial IDs.
- Source model ID, scaler ID, dataset ID, and adapter/generator version.
- Mahalanobis chi-squared threshold derived from latent dimensionality and confidence level.

Acceptance criteria:

- Known B0 synthetic faults produce repeatable clusters.
- Replayed known faults match the dictionary more often than the withheld novel fault.
- The withheld novel fault is not silently forced into an existing dictionary entry.

Deliverable:

- A dictionary artifact and evaluation output showing known/novel decisions.

### Phase 5: Public Dataset Adapter

Add one public dataset adapter after the synthetic path works.

Recommended sequence:

1. Paderborn adapter, because it includes current and vibration signals.
2. CWRU adapter only if a simpler vibration-only bearing check is useful.

The adapter should convert external files into canonical raw trial folders with `source_type: external` and a source citation in `manifest.yaml`. If channels are missing relative to the 2109 profile, use a separate dataset config rather than filling fake channels into final-schema experiments.

Implementation note, 2026-05-17: the runnable public-data path currently uses CWRU as `source_type: external_cwru`. It downloads selected Zenodo-hosted CWRU `.mat` files, writes canonical raw trial folders, and uses a separate 1200-dimensional vibration-only dataset config. Paderborn remains the preferred final public source because it includes motor current and vibration, but its official archives are large RAR files and are not the default smoke/public check.

Acceptance criteria:

- The same `make-dataset`, `train-sdae`, `build-dictionary`, and `evaluate` commands run on external data.
- Any reduced channel profile is explicit in the dataset manifest.
- Results are labelled as proof-of-concept only.

Deliverable:

- One public-data adapter and a short external-data evaluation report.

### Phase 6: FedRep and DANN Dry Run

Use the synthetic baselines as simulated clients/domains.

FedRep dry run:

- Treat B0-B4 as five clients.
- Train/update the shared encoder and local decoders.
- Regenerate the fault dictionary after encoder updates.
- Evaluate whether known B0 fault dictionary entries remain valid on B1-B4.

DANN dry run:

- Use healthy B0-B4 windows with domain labels.
- Pretrain the SDAE.
- Train the DANN feature extractor with the domain classifier branch.
- Discard the classifier for deployment.
- Rebuild the dictionary and run the same cross-domain evaluation.

Acceptance criteria:

- Both workflows run through the shared artifact and evaluation system.
- Cross-domain accuracy, centroid drift, and DANN domain confusion metrics are produced.
- The runtime diagnostic path remains identical after training: window -> model -> reconstruction error -> latent buffer -> HDBSCAN -> dictionary.

Deliverable:

- Comparable proof-of-concept metric tables for baseline SDAE, FedRep, and DANN.

### Phase 7: Replay Runtime

Run synthetic and external data through the same `run` mode intended for live edge inference:

```powershell
usv-faults run --source replay --trial data/raw/trials/2026-05-14_POC_B0_fault_bearing_T001 --model artifacts/models/run_poc_sdae_0001 --dictionary artifacts/dictionaries/dict_poc_b0_0001
```

Acceptance criteria:

- The pipeline produces timestamped anomaly and isolation decisions.
- Runtime logs contain reconstruction error, threshold state, cluster labels, dictionary decisions, CPU time, and memory use.
- The replay path can later be swapped for the Teensy serial source without changing model code.

Deliverable:

- A replay log that looks like a future live-edge diagnostic log.

## 6. Proof-of-Concept Outputs

The POC should produce the following artifacts:

```text
configs/
  poc_synthetic.yaml
  dataset_poc_synthetic.yaml
data/raw/trials/
  2026-05-14_POC_...
data/processed/datasets/
  ds_poc_synthetic_0001/
artifacts/models/
  run_poc_sdae_0001/
artifacts/dictionaries/
  dict_poc_b0_0001/
runs/reports/
  poc_detection_metrics.csv
  poc_isolation_metrics.csv
  poc_cross_domain_metrics.csv
  poc_summary.md
```

The final `poc_summary.md` should state clearly which results came from synthetic data and which came from public data.

## 7. Definition of Done

The proof of concept is complete when:

- Synthetic data can be attached as raw trial folders using the same manifest structure as real recordings.
- Dataset generation, SDAE training, threshold selection, HDBSCAN clustering, dictionary creation, and replay evaluation all run through shared commands.
- The code path for synthetic data and future hardware data differs only at the source adapter.
- At least two known synthetic faults are detected and assigned repeatable dictionary entries.
- At least one withheld synthetic fault is handled as a novel fault.
- B1-B4 synthetic baseline shifts produce cross-domain metrics for baseline SDAE, and preferably for FedRep and DANN dry runs.
- One public dataset adapter has been tested or explicitly deferred with a reason.
- All artifacts record enough metadata to reproduce the run.

## 8. Risks and Controls

- Synthetic data can be too clean. Add stochastic noise, transient shocks, baseline drift, and benign impulses early.
- Synthetic data can overfit the expected result. Keep public data as a realism check and keep final thesis claims tied to self-collected data.
- Public datasets may not match the 2109 input schema. Use separate configs for reduced channel profiles and do not mix them with final-schema results.
- The 2109-dimensional profile conflicts with a possible six-axis accelerometer setup. Resolve the final channel count before locking model dimensions.
- HDBSCAN may be unstable in a 420-dimensional latent space. Keep latent dimension and HDBSCAN parameters configurable from the first implementation.
- FedRep and DANN may distract from the baseline. Do not start them until baseline SDAE, HDBSCAN, and dictionary artifacts exist.

## 9. Current Proof-of-Concept Implementation Notes

These notes reflect the first working POC implementation pass on 2026-05-14.

- Milestones 1-3 have been smoke-verified, not full-matrix verified. The smoke datasets are intentionally small and are not thesis evidence.
- Use `configs/poc_synthetic_training_smoke.yaml`, `configs/dataset_poc_synthetic_training_smoke.yaml`, and `configs/baseline_sdae_smoke.yaml` for quick training checks.
- Use `configs/poc_synthetic.yaml`, `configs/dataset_poc_synthetic.yaml`, and `configs/baseline_sdae.yaml` for fuller POC runs once runtime and disk usage are acceptable.
- `telemetry.parquet` is the synthetic raw signal file. `events.csv` is only an event timeline and is expected to contain only a few rows.
- `attach-data` should not rewrite existing raw trials. Existing canonical trial files should be reused and quality-checked, matching the immutable raw data rule.
- The SDAE activation policy should be explicit in config: hidden layers use `relu`, the reconstruction output uses `sigmoid`, and the latent layer remains linear.
- The current smoke SDAE uses `2109 -> 128 -> 64 -> 16 -> 64 -> 128 -> 2109` so tests run quickly. The planned full SDAE remains `2109 -> 2048 -> 1024 -> 420 -> 1024 -> 2048 -> 2109`.
- The current scaler is fitted only on healthy training windows and saved under the planned name `scaler.joblib`, but the implementation uses pickle to avoid requiring the external `joblib` package.
- The current plots are limited to `loss_curve.png` and `reconstruction_error_hist.png`; they are artifacts for training sanity checks, not model or latent visualisations.

## Attachment A: Simulation Data Profile

The companion YAML file `poc_synthetic_data_attachment.yaml` defines the initial synthetic-data attachment profile. It should be treated as a draft config for `configs/poc_synthetic.yaml` once implementation begins.

The attachment's role is to specify the simulated trials and channel assumptions only. It should not contain model logic, training logic, or evaluation logic.
