# USV Faults Proof of Concept

This package implements the proof-of-concept pipeline described in the thesis implementation plans.

## Local Environment

Create the local environment:

```powershell
.\scripts\setup_env.ps1
```

This creates `.venv` with `--system-site-packages` so the project can reuse the numerical and parquet libraries already installed on this machine. The package now requires the numerical stack used by the thesis pipeline, including PyTorch, HDBSCAN, scikit-learn, SciPy, joblib, and Matplotlib.

On Windows/Python 3.9, `hdbscan==0.8.40` is pinned because that version has a prebuilt wheel in this environment. Newer HDBSCAN releases may try to compile from source and require Microsoft C++ Build Tools.

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
rolling_window_size: 300
min_cluster_size: 15
min_samples: 15
metric: euclidean
cluster_selection_method: eom
allow_single_cluster: true
mahalanobis_confidence: 0.99
dictionary_baseline_id: 0
known_fault_labels: [bearing_impulse, propeller_imbalance]
withheld_fault_labels: [shaft_rub]
```

For smoke training, the SDAE latent dimension is `16`, so the Mahalanobis known/novel threshold is `chi2.ppf(0.99, 16) = 31.9999`. For the thesis-default latent dimension of `420`, the same rule gives the planned threshold near `487.6`.

The Objective 4 tests verify the statistical pieces directly and then run a reduced integration path through synthetic data, dataset generation, SDAE training, HDBSCAN clustering, Ledoit-Wolf dictionary creation, and artifact writing.

The first runnable objective is synthetic data attachment and quality checking:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
```

Synthetic data is written through the same raw-trial folder contract intended for future hardware data, so downstream preprocessing and training can reuse the same path.
