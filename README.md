# USV Faults Proof of Concept

This package implements the proof-of-concept pipeline described in the thesis implementation plans.

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
min_runtime_cluster_size: 15
cluster_match_min_member_fraction: 0.50
dictionary_baseline_id: 0
known_fault_labels: [bearing_impulse, propeller_imbalance]
withheld_fault_labels: [shaft_rub]
```

For smoke training, the SDAE latent dimension is `16`, so the Mahalanobis known/novel threshold is `chi2.ppf(0.99, 16) = 31.9999`. For the thesis-default latent dimension of `420`, the same rule gives the planned threshold near `487.6`.

The Objective 4 tests verify the statistical pieces directly and then run a reduced integration path through synthetic data, dataset generation, SDAE training, HDBSCAN clustering, Ledoit-Wolf dictionary creation, and artifact writing.

Run the objective 5 checks:

```powershell
.\scripts\run_objective_5_checks.ps1
```

Objective 5 evaluates a trained model and dictionary against a processed dataset:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli evaluate --model artifacts/models/run_poc_sdae_smoke_objective_5 --dictionary artifacts/dictionaries/dict_poc_b0_smoke_objective_5 --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --out runs/reports/objective_5_smoke
```

It writes:

```text
poc_detection_metrics.csv
poc_isolation_metrics.csv
poc_cross_domain_metrics.csv
poc_performance_metrics.csv
poc_window_decisions.csv
poc_summary.md
```

`poc_performance_metrics.csv` records Objective 5 performance indicators for the trained SDAE artifact. It includes parameter count, estimated forward FLOPs per 100 ms window, estimated training FLOPs, measured training CPU/RAM from `train-sdae`, and measured offline inference CPU/RAM/latency from `evaluate`.

Objective 5 also adds replay runtime logging from a raw trial folder:

```powershell
.\.venv\Scripts\python.exe -m usv_faults.cli run --source replay --trial data/raw/trials_training_smoke/2026-05-14_POC_B0_fault_bearing_T001 --model artifacts/models/run_poc_sdae_smoke_objective_5 --dictionary artifacts/dictionaries/dict_poc_b0_smoke_objective_5 --out runs/logs/objective_5_smoke
```

Dictionary decisions now use rolling cluster matching. The runtime path keeps the last 30 latent windows, clusters the anomalous latents inside that temporal buffer with HDBSCAN, then compares the current runtime cluster centroid and member inlier fraction against stored dictionary clusters. Replay logs contain reconstruction error, threshold state, runtime cluster label, dictionary decision, matched fault ID/label, cluster support count, member inlier fraction, and squared Mahalanobis centroid distance.

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

The first runnable objective is synthetic data attachment and quality checking:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
```

Synthetic data is written through the same raw-trial folder contract intended for future hardware data, so downstream preprocessing and training can reuse the same path.
