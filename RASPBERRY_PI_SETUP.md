# Raspberry Pi Container Setup

This guide explains how to run the USV faults proof of concept on a Raspberry Pi using the same Linux container workflow used during Windows development.

The container is intentionally a full training/evaluation environment. It is not an inference-only image. The Pi can run SDAE training, HDBSCAN dictionary generation, Mahalanobis/Ledoit-Wolf decisions, evaluation, and replay commands inside the container. Future FedRep/DANN training commands should use the same container path.

## 1. Hardware and OS Assumptions

Recommended minimum:

- Raspberry Pi 5, or Raspberry Pi 4 with enough patience
- 64-bit Raspberry Pi OS, Debian, or Ubuntu Server
- 8 GB RAM preferred for full training experiments
- 32 GB microSD minimum, 64 GB or larger recommended
- active cooling for longer training runs
- network access during the first Docker build

Check the Pi is running a 64-bit OS:

```bash
uname -m
getconf LONG_BIT
dpkg --print-architecture
```

Expected values are typically:

```text
aarch64
64
arm64
```

If the Pi is running 32-bit userland, do not use it for the full thesis training container. Reinstall a 64-bit OS first.

## 2. Install Docker Engine

Follow Docker's official Debian installation guide for the current commands:

```text
https://docs.docker.com/engine/install/debian/
```

For a 64-bit Debian/Raspberry Pi OS-style install, the command shape is:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify Docker:

```bash
sudo docker run hello-world
docker compose version
```

Docker's Linux post-install guide covers non-root Docker usage:

```text
https://docs.docker.com/engine/install/linux-postinstall/
```

To run Docker without `sudo`:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker run hello-world
```

Membership in the `docker` group effectively grants root-equivalent control of Docker on that host. This is convenient for thesis bench testing, but it is still a security decision.

## 3. Copy the Project to the Pi

Option A: clone the repository on the Pi:

```bash
git clone <your-repository-url> ~/usv-faults
cd ~/usv-faults
```

Option B: copy the `Implementation` folder from Windows to the Pi:

```powershell
scp -r C:\Users\jacks\OneDrive\Thesis\Implementation pi@<pi-hostname-or-ip>:~/usv-faults
```

Then on the Pi:

```bash
cd ~/usv-faults
```

Do not copy large generated folders unless you intentionally need them:

```text
data/
artifacts/
runs/
.venv/
```

They can be regenerated inside the Pi container.

## 4. Build the Container on the Pi

Build the normal image:

```bash
bash scripts/docker_build.sh
```

The default build installs CPU-only PyTorch from:

```text
https://download.pytorch.org/whl/cpu
```

This avoids the very large CUDA package downloads that are not useful on Raspberry Pi.

If the Pi ARM64 build cannot resolve a CPU PyTorch wheel from that index, fall back to normal PyPI resolution:

```bash
TORCH_INDEX_URL= bash scripts/docker_build.sh
```

The first build can take a long time because it downloads and installs PyTorch, SciPy, scikit-learn, HDBSCAN, Pandas, PyArrow, and Matplotlib. Keep the Pi powered and cooled.

## 5. Run the Smoke Checks

Run the full container check:

```bash
bash scripts/docker_test.sh
```

This runs:

```text
python -m unittest discover -s tests
python -m usv_faults.cli --help
Objective 1 smoke path
Objective 2 smoke path
Objective 3 smoke path
Objective 4 smoke path
Objective 5 smoke path
```

Run individual smoke paths:

```bash
bash scripts/run_objective_1_checks.sh
bash scripts/run_objective_2_checks.sh
bash scripts/run_objective_3_checks.sh
bash scripts/run_objective_4_checks.sh
bash scripts/run_objective_5_checks.sh
```

Or run directly through Docker Compose:

```bash
docker compose run --rm usv-faults bash scripts/run_container_checks.sh
```

## 6. Open a Container Shell

```bash
bash scripts/docker_shell.sh
```

Inside the container:

```bash
python -m usv_faults.cli --help
python -m unittest discover -s tests
```

The repository is mounted at:

```text
/app
```

Generated `data/`, `artifacts/`, and `runs/` folders appear in the host project folder on the Pi.

## 7. Run the Objective 5 Pipeline on the Pi

```bash
docker compose run --rm usv-faults bash scripts/run_objective_5_checks.sh
```

Expected key outputs:

```text
data/processed/datasets/ds_poc_synthetic_training_smoke/
artifacts/models/run_poc_sdae_smoke_objective_5/
artifacts/dictionaries/dict_poc_b0_smoke_objective_5/
runs/reports/objective_5_smoke/
runs/logs/objective_5_smoke/
```

## 8. Serial/Teensy Access Later

When live collection is implemented, identify the serial device:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

Give the Pi user serial access:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in, then run the container with the serial device mounted:

```bash
docker compose run --rm --device /dev/ttyACM0:/dev/ttyACM0 usv-faults bash
```

The current implementation supports replay from raw trial folders. Live Teensy collection is still a later integration step.

## 9. Performance Notes

The Windows Docker build was verified with:

```text
torch 2.8.0+cpu
CUDA unavailable
Objective 1-5 smoke checks passed
8 unit tests passed
```

The Pi must still be measured directly for:

- training time
- inference latency
- CPU usage
- RAM usage
- power draw
- thermal throttling
- serial timing

Containerisation makes the software environment reproducible. It does not make Windows performance evidence equivalent to Raspberry Pi performance evidence.

## 10. Cleanup Commands

Show Docker disk usage:

```bash
docker system df
```

Remove stopped containers and unused layers:

```bash
docker system prune
```

Remove generated experiment outputs if you intentionally want a clean project run:

```bash
rm -rf data artifacts runs
```

Do not run cleanup commands unless you are sure you no longer need the generated artifacts.

