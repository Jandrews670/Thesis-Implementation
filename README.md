# USV Faults Proof of Concept

This package implements the proof-of-concept pipeline described in the thesis implementation plans.

## Local Environment

Create the local environment:

```powershell
.\scripts\setup_env.ps1
```

This creates `.venv` with `--system-site-packages` so the project can reuse the numerical and parquet libraries already installed on this machine. Objective 1 intentionally uses stdlib `argparse` and local schema classes rather than requiring extra CLI/schema packages.

Run the objective 1 checks:

```powershell
.\scripts\run_objective_1_checks.ps1
```

The check script uses `configs/poc_synthetic_smoke.yaml` so it validates the raw-trial path quickly. Use `configs/poc_synthetic.yaml` when you intentionally want the full proof-of-concept synthetic dataset.

The first runnable objective is synthetic data attachment and quality checking:

```powershell
usv-faults attach-data --source synthetic --config configs/poc_synthetic.yaml --out data/raw/trials
usv-faults qc --trial data/raw/trials/2026-05-14_POC_B0_nominal_T001
```

Synthetic data is written through the same raw-trial folder contract intended for future hardware data, so downstream preprocessing and training can reuse the same path.
