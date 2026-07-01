# USV Faults Proof of Concept

This package implements the thesis proof-of-concept pipeline for USV fault detection and isolation. The implemented baseline is a sparse denoising autoencoder (SDAE) for healthy-behaviour modelling, HDBSCAN latent fault grouping, and a Ledoit-Wolf/Mahalanobis dictionary for known/novel fault decisions.

## Guide for Markers

The repository is organised so the implementation can be inspected without needing generated datasets or local planning notes.

| Path | Purpose |
|---|---|
| `src/usv_faults/` | Main Python package. This contains source adapters, raw-trial storage, preprocessing/windowing, SDAE training, HDBSCAN/Mahalanobis dictionary construction, evaluation, replay, and performance measurement. |
| `configs/` | Reproducible YAML configs for synthetic data, public bearing datasets, SDAE model shapes, HDBSCAN/dictionary parameters, and known/withheld fault labels. |
| `tests/` | Unit and smoke tests for the implemented objectives, including synthetic data, 2109-D windowing, SDAE artifact creation, dictionary generation, evaluation reports, replay logs, and public bearing adapters. |
| `scripts/` | PowerShell/Bash helpers for setup, objective smoke checks, Docker build/test/shell commands, public dataset checks, Mahalanobis sweeps, and FEMTO degradation analysis. |
| `Dockerfile` and `docker-compose.yml` | Linux container path intended to make Windows development and Raspberry Pi deployment use the same Python/PyTorch/HDBSCAN environment. |
| `README.md` | This high-level repository guide and command reference. |
| `USER_GUIDE.md` | More detailed user workflow documentation. |
| `RASPBERRY_PI_SETUP.md` | Raspberry Pi Docker setup and deployment checklist. |
| `data/` | Generated raw and processed datasets. Ignored by git because these can be large and can be regenerated. |
| `artifacts/` | Generated trained models, scalers, thresholds, dictionaries, plots, and manifests. Ignored by git. |
| `runs/` | Generated evaluation reports, sweeps, replay logs, and local run outputs. Ignored by git. |

Local planning notes and agent handoff files have been moved to `internal_notes/`, which is ignored by git. They are not required to run or assess the implementation.

The main implementation flow is:

```text
attach-data -> qc/preview -> make-dataset -> train-sdae -> build-dictionary -> evaluate/run
```

The core CLI is:

```powershell
.\.venv\Scripts\python.exe -B -m usv_faults.cli --help
```

## Local Environment

Create the local environment:

```powershell
.\scripts\setup_env.ps1
```

This creates `.venv` with `--system-site-packages` so the project can reuse the numerical and parquet libraries already installed on this machine. The package now requires the numerical stack used by the thesis pipeline, including PyTorch, HDBSCAN, scikit-learn, SciPy, joblib, and Matplotlib.

On Windows/Python 3.9, `hdbscan==0.8.40` is pinned because that version has a prebuilt wheel in this environment. Newer HDBSCAN releases may try to compile from source and require Microsoft C++ Build Tools.

## Docker and Raspberry Pi Linux

The recommended Raspberry Pi path is to run the same Linux container during Windows development and on the Pi. The image is intentionally a full training image: it includes PyTorch, HDBSCAN, SciPy, scikit-learn, Matplotlib, build tools, OpenBLAS/LAPACK, and the package itself. This keeps the current SDAE training, dictionary generation, evaluation, and future FedRep/DANN training paths inside one reproducible Linux environment.

For the full Pi setup checklist, see [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md).

Start Docker Desktop first, then build and test from Windows:

```powershell
.\scripts\docker_build.ps1
.\scripts\docker_test.ps1
```

The default Docker build installs CPU-only PyTorch from `https://download.pytorch.org/whl/cpu`. That avoids the very large CUDA dependency downloads that are unnecessary for Raspberry Pi. If a particular ARM64/Pi Python environment cannot resolve the CPU index wheel, fall back to the normal PyPI Torch resolver:

```powershell
.\scripts\docker_build.ps1 -TorchIndexUrl ""
```

Open a shell in the container:

```powershell
.\scripts\docker_shell.ps1
```

Build an ARM64 image from Windows for Raspberry Pi compatibility testing:

```powershell
.\scripts\docker_build.ps1 -Platform linux/arm64 -Tag usv-faults:pi
```

On the Raspberry Pi, install Docker Engine, clone or copy this repository, then run:

```bash
bash scripts/docker_build.sh
bash scripts/docker_test.sh
```

The short Pi checklist is:

```bash
uname -m
getconf LONG_BIT
dpkg --print-architecture
docker compose version
cd ~/usv-faults
bash scripts/docker_build.sh
bash scripts/docker_test.sh
```

Linux fallback to PyPI Torch resolution:

```bash
TORCH_INDEX_URL= bash scripts/docker_build.sh
```

The Linux/container smoke checks use Bash scripts:

```bash
bash scripts/run_container_checks.sh
bash scripts/run_objective_5_checks.sh
```

For future live Teensy access, run the container with the serial device mounted, for example:

```bash
docker compose run --rm --device /dev/ttyACM0:/dev/ttyACM0 usv-faults bash
```

Containerisation makes the Python/Linux software environment reproducible, but it does not remove the need to validate Pi-specific serial permissions, CPU/RAM/power measurements, and hardware timing on the actual Raspberry Pi.

Run the objective 1 checks:

```powershell
.\scripts\run_objective_1_checks.ps1
```

The check script uses `configs/poc_synthetic_smoke.yaml` so it validates the raw-trial path quickly. Use `configs/poc_synthetic.yaml` when you intentionally want the full proof-of-concept synthetic dataset.

Run the objective 2 checks:

```powershell
.\scripts\run_objective_2_checks.ps1
```

Objective 2 adds telemetry previews and processed 100 ms windows:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli preview --trial data/raw/trials_smoke/2026-05-14_POC_B0_nominal_T001
.\.venv\Scripts\python.exe -m usv_faults.cli make-dataset --config configs/dataset_poc_synthetic_smoke.yaml --out data/processed/datasets/ds_poc_synthetic_smoke
```

Run the objective 3 checks:

```powershell
.\scripts\run_objective_3_checks.ps1
```

Objective 3 trains the baseline SDAE on healthy windows and writes a model artifact:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --config configs/baseline_sdae_smoke.yaml --out artifacts/models/run_poc_sdae_smoke
```

Run the objective 4 checks:

```powershell
.\scripts\run_objective_4_checks.ps1
```

Objective 4 extracts SDAE latent vectors, runs `hdbscan.HDBSCAN`, and writes a Ledoit-Wolf/Mahalanobis fault dictionary:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli build-dictionary --model artifacts/models/run_poc_sdae_smoke --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --config configs/hdbscan.yaml --out artifacts/dictionaries/dict_poc_b0_smoke
```

The default dictionary parameters are:

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
known_fault_labels: [bearing_impulse, propeller_imbalance]
withheld_fault_labels: [shaft_rub]
```

For smoke training, the SDAE latent dimension is `16`, so the theoretical Mahalanobis known/novel threshold is `chi2.ppf(0.99, 16) = 31.9999`. When `mahalanobis_empirical_enabled` is true, each dictionary entry uses the tighter of that chi-square threshold and its own source-cluster empirical radius. Set `mahalanobis_empirical_enabled: false` to restore the chi-square-only decision gate.

The Objective 4 tests verify the statistical pieces directly and then run a reduced integration path through synthetic data, dataset generation, SDAE training, HDBSCAN clustering, Ledoit-Wolf dictionary creation, and artifact writing.

Run the objective 5 checks:

```powershell
.\scripts\run_objective_5_checks.ps1
```

Objective 5 evaluates a trained model and dictionary against a processed dataset:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli evaluate --model artifacts/models/run_poc_sdae_smoke_objective_5 --dictionary artifacts/dictionaries/dict_poc_b0_smoke_objective_5 --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --out runs/reports/objective_5_smoke
```

Evaluation metrics exclude the first 10 windows of each contiguous trial/baseline/fault state by default. This avoids scoring short startup or state-transition transients now that event-level voting is the main reporting layer. The raw `poc_window_decisions.csv` and `poc_event_decisions.csv` files still contain every window and include `state_window_index`, `metric_excluded`, and `metric_exclusion_reason` columns. Use `--metric-warmup-windows 0` to reproduce the old no-skip metric behaviour.

It writes:

```text
poc_detection_metrics.csv
poc_isolation_metrics.csv
poc_event_metrics.csv
poc_cross_domain_metrics.csv
poc_performance_metrics.csv
poc_window_decisions.csv
poc_event_decisions.csv
poc_summary.md
```

`poc_performance_metrics.csv` records Objective 5 performance indicators for the trained SDAE artifact. It includes parameter count, estimated forward FLOPs per 100 ms window, estimated training FLOPs, measured training CPU/RAM from `train-sdae`, and measured offline inference CPU/RAM/latency from `evaluate`.

`poc_window_decisions.csv` remains the per-window SDAE/HDBSCAN/Mahalanobis decision log. `poc_event_decisions.csv` adds a rolling event layer that votes over recent window decisions, and `poc_event_metrics.csv` reports event-level false positive, detection, known-fault isolation, withheld-novel, and latency metrics.

Latest public-dataset Mahalanobis sweeps are generated under `runs/reports/mahalanobis_confidence_sweep/`. The current evidence compares chi-square-only thresholds against empirical `p=0.74` thresholds across CWRU, expanded IMS, and FEMTO with the 10-window metric warmup active. The empirical `p=0.74` setting fixes the IMS withheld-novel failure case while preserving strong CWRU withheld-novel behavior.

Objective 5 also adds replay runtime logging from a raw trial folder:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli run --source replay --trial data/raw/trials_training_smoke/2026-05-14_POC_B0_fault_bearing_T001 --model artifacts/models/run_poc_sdae_smoke_objective_5 --dictionary artifacts/dictionaries/dict_poc_b0_smoke_objective_5 --out runs/logs/objective_5_smoke
```

Dictionary decisions now use rolling cluster matching. The runtime path keeps a rolling latent window, clusters the anomalous latents inside that temporal buffer with HDBSCAN, then compares the current runtime cluster centroid and member inlier fraction against stored dictionary clusters. Event reports then smooth these per-window decisions with a separate rolling vote. Replay logs contain reconstruction error, threshold state, runtime cluster label, dictionary decision, matched fault ID/label, cluster support count, member inlier fraction, and squared Mahalanobis centroid distance.

Run the objective 7 public-data check:

```powershell
.\scripts\run_objective_7_public_checks.ps1
```

Objective 7 attaches selected public CWRU bearing `.mat` files, converts them into canonical raw trial folders, and runs the current Objective 2-5 path on a reduced vibration-only profile:

```text
data/raw/public_cwru
data/processed/datasets/ds_public_cwru_objective_7
artifacts/models/run_public_cwru_sdae_objective_7
artifacts/dictionaries/dict_public_cwru_objective_7
runs/reports/objective_7_public_cwru
```

The CWRU config uses one 12 kHz drive-end vibration channel, 100 ms windows, and `expected_input_dim: 1200`. It does not fabricate missing current channels to match the 2109-D USV schema. The generated public report is labelled as a public CWRU realism check, not final thesis evidence.

Additional public bearing adapters are available for local dataset experiments:

```text
IMS/NASA Bearings: configs/public_ims.yaml, dataset_public_ims.yaml
FEMTO/PRONOSTIA: configs/public_femto.yaml, dataset_public_femto.yaml
HUST Bearings: configs/public_hust.yaml, dataset_public_hust.yaml
Paderborn Bearings: configs/public_paderborn.yaml, dataset_public_paderborn.yaml
```

These adapters convert downloaded/extracted public files into the same canonical raw-trial folder contract as CWRU and synthetic data. They are template configs because the public archives can be large and their extracted folder names vary. Update the `path`, `records`, `columns`, or `mat_variables` fields after downloading the data locally.

Run the adapter fixture checks:

```powershell
.\scripts\run_public_bearing_adapter_checks.ps1
```

Example IMS command sequence after placing the files under `data/external/ims`:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli attach-data --source ims --config configs/public_ims.yaml --out data/raw/public_ims
.\.venv\Scripts\python.exe -m usv_faults.cli make-dataset --config configs/dataset_public_ims.yaml --out data/processed/datasets/ds_public_ims
.\.venv\Scripts\python.exe -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_public_ims --config configs/baseline_sdae_public_ims.yaml --out artifacts/models/run_public_ims_sdae
.\.venv\Scripts\python.exe -m usv_faults.cli build-dictionary --model artifacts/models/run_public_ims_sdae --dataset data/processed/datasets/ds_public_ims --config configs/hdbscan_public_ims.yaml --out artifacts/dictionaries/dict_public_ims
.\.venv\Scripts\python.exe -m usv_faults.cli evaluate --model artifacts/models/run_public_ims_sdae --dictionary artifacts/dictionaries/dict_public_ims --dataset data/processed/datasets/ds_public_ims --out runs/reports/public_ims
```

The first runnable objective is synthetic data attachment and quality checking:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
```

Synthetic data is written through the same raw-trial folder contract intended for future hardware data, so downstream preprocessing and training can reuse the same path.
