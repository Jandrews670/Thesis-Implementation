# High-Level Implementation Plan

This implementation should stay close to the structure already defined in the initial report: build the baseline two-stage fault detection system first, validate it against simple test-rig data, then add the FedRep and DANN variants only after the core data path is stable. The main objective is not to build a large software platform; it is to produce a reproducible experimental pipeline that can run on the Raspberry Pi 5, train on collected bench-top data, and generate the metrics needed for the final thesis comparison.

## 1. System Shape

The software should be organised around one continuous data path:

```text
Teensy 4.1 sensors
  -> USB telemetry stream
  -> Raspberry Pi receiver
  -> calibrated samples
  -> 100 ms sliding windows
  -> SDAE reconstruction + latent vector
  -> reconstruction-error fault trigger
  -> HDBSCAN over recent latent vectors
  -> Mahalanobis/Ledoit-Wolf fault dictionary
  -> known fault, novel fault, or healthy state
```

During normal operation, the Raspberry Pi should do only the work that must happen on the edge: receive telemetry, construct windows, run inference, maintain the rolling latent buffer, and trigger clustering when needed. Heavier training, hyperparameter sweeps, DANN training, and most FedRep simulation should be run on a development machine or central server, then exported back to the Pi.

The first implementation should support three execution modes:

- `collect`: receive and store telemetry from the Teensy/test rig.
- `train`: train or tune models from saved datasets.
- `run`: replay a dataset or run live edge inference using a trained model and fault dictionary.

This keeps the hardware work, ML work, and evaluation work connected without forcing every experiment to be run live.

## 2. Recommended Libraries

Use Python for the Raspberry Pi and training pipeline unless a specific bottleneck proves otherwise.

- Core numerical stack: `numpy`, `scipy`, `pandas` or `polars`.
- Model training and inference: `torch`.
- Optional deployment/export: `torch.jit` first; consider `onnx` and `onnxruntime` if PyTorch CPU inference is too heavy on the Pi.
- Clustering: `hdbscan`.
- Classical ML utilities: `scikit-learn`, especially `StandardScaler`, `LedoitWolf`, train/test splitting, and metrics.
- Data storage: start with `parquet` via `pyarrow` for processed windows and metadata; use `h5py` or `zarr` if high-rate raw arrays become awkward in tabular form.
- USB/serial ingestion: `pyserial`; use a dedicated reader thread or `asyncio` wrapper once the framing format is fixed.
- CLI and configuration: `typer`, `pydantic`, and YAML config files.
- Experiment tracking: lightweight CSV/JSON summaries at first; add `MLflow` or TensorBoard only if manual run tracking becomes messy.
- Plotting/report figures: `matplotlib`, `seaborn`, and optionally `plotly` for exploratory plots.
- Testing and code quality: `pytest`, `ruff`, and type hints for core data structures.
- Teensy firmware: Arduino/PlatformIO C++ with hardware timer/DMA sampling and binary USB serial packets.

Avoid starting with a database, web dashboard, or complex distributed framework. They are not on the critical path.

## 3. Suggested Repository Layout

When implementation begins, structure the code as a small Python package rather than loose notebooks:

```text
implementation/
  implementation_plan.md
  pyproject.toml
  README.md
  configs/
    baseline_sdae.yaml
    hdbscan.yaml
    fedrep.yaml
    dann.yaml
    acquisition.yaml
  firmware/
    teensy_daq/
  src/
    usv_faults/
      daq/
        serial_reader.py
        packet_format.py
        calibration.py
      preprocessing/
        windowing.py
        feature_scaling.py
        datasets.py
      models/
        sdae.py
        dann.py
        fedrep.py
        export.py
      clustering/
        hdbscan_pipeline.py
        fault_dictionary.py
        mahalanobis.py
      training/
        train_sdae.py
        train_dann.py
        train_fedrep.py
        threshold_search.py
      evaluation/
        metrics.py
        trial_runner.py
        plots.py
      edge/
        realtime_pipeline.py
        model_service.py
      server/
        federated_averaging.py
        artifact_store.py
  data/
    raw/
    processed/
    manifests/
  runs/
  tests/
```

The important boundary is that `daq` only knows about raw samples and timestamps, `preprocessing` turns samples into model-ready windows, `models` only defines neural networks, `clustering` owns fault identification, and `evaluation` owns thesis metrics. This separation will make it much easier to compare SDAE-only, FedRep, and DANN variants without rewriting the pipeline.

## 4. Implementation Phases

### Phase 0: Foundation Before Hardware Is Ready

This should be done immediately while the physical test setup is being procured and assembled.

- Create the Python package, configs, CLI entry points, and test skeleton.
- Define the canonical telemetry schema: timestamp, motor current, accelerometer channels, voltage, temperature, PWM command, baseline label, trial ID, and fault label when known.
- Create a synthetic/replay data source so the ML pipeline can be developed before live data exists.
- Implement deterministic windowing for the report's initial 100 ms window size.
- Implement preprocessing that preserves the report's input assumptions: high-frequency vibration retained at 10 kHz, current decimated to 1 kHz, and scalar/environmental values represented by mean, variance, and peak-to-peak values.
- Validate that the processed input vector is dimensioned consistently with the planned 2109-dimensional SDAE input.

Deliverable: a working offline command that can turn either synthetic data or recorded raw data into processed training windows.

### Phase 1: Data Acquisition and Test-Rig Integration

The Teensy should own precise sampling. The Pi should not try to directly sample high-rate analog data under Linux.

- Implement Teensy firmware for hardware-timed acquisition of accelerometer and current channels at 10 kHz.
- Stream binary packets over high-speed USB with sequence numbers, timestamps, and a checksum/CRC.
- Add Pi-side packet decoding with loss detection and logging.
- Store raw telemetry without modification first; all calibration and feature generation should be reproducible from raw files.
- Add a live ring buffer on the Pi for the latest samples and a separate writer path for persistent recording.
- Create a trial manifest file for each run, recording baseline condition, motor command profile, induced fault type, sensor configuration, and notes.

Deliverable: repeatable healthy baseline recordings for Baseline 0, plus short sanity-check recordings for the other environmental baselines.

### Phase 2: Baseline SDAE Fault Detection

Start with the simplest version: one Sparse Denoising Autoencoder trained on healthy data only, with no FedRep or DANN augmentation.

Initial architecture from the report:

```text
input: 2109
encoder hidden: 2048 -> 1024
latent: 420
decoder hidden: 1024 -> 2048
output: 2109
```

Training details:

- Use masking noise during training, initially zeroing 30 percent of input dimensions.
- Use reconstruction MSE plus an L1 activation penalty on the latent representation.
- Normalise features using parameters fitted only on healthy training data.
- Use validation data from held-out healthy windows to set a reconstruction-error threshold.
- Treat threshold selection as an experimental variable, not a hard-coded constant.
- Track false positive rate and true fault detection rate before adding HDBSCAN.

Deliverable: a trained baseline SDAE that can flag anomalous windows from recorded data and live replay.

### Phase 3: HDBSCAN Fault Isolation and Fault Dictionary

Once reconstruction error can reliably trigger a fault state, add the second stage.

- Maintain a rolling buffer of latent vectors for the last 300 windows, matching the report's 30-second aggregation target.
- When the reconstruction threshold is exceeded persistently, run HDBSCAN over the rolling latent buffer.
- Tune `min_cluster_size`, `min_samples`, and distance settings empirically.
- Compute cluster centroids and covariance estimates for non-healthy clusters.
- Use `sklearn.covariance.LedoitWolf` to keep covariance matrices invertible when cluster sizes are small relative to latent dimensionality.
- Use squared Mahalanobis distance for known/novel fault decisions.
- Compute the chi-squared threshold from the latent dimensionality and desired confidence rather than hard-coding it; for the initial 420-dimensional latent space, the report's 99 percent threshold is approximately 487.6.
- Store each fault dictionary entry with centroid, covariance/shrinkage parameters, sample count, model version, scaler version, baseline source, and human-readable label if available.

Deliverable: a full SDAE + HDBSCAN + fault dictionary pipeline that can identify known fault clusters or declare a novel fault.

### Phase 4: Experiment Harness and Metrics

The experiment harness should be built before large-scale data collection; otherwise, it will be difficult to keep trials comparable.

- Add a single trial runner that can execute live trials or replay recorded datasets.
- Record every model artifact, config file, threshold, data manifest, and software version used in a run.
- Calculate the thesis metrics directly from saved outputs:
- False positive rate.
- True fault detection rate.
- True fault isolation rate.
- Fault isolation latency.
- DBCV score.
- Cross-domain accuracy.
- Maximum centroid drift distance.
- DANN domain confusion rate.
- SWaP-C metrics: model update size, peak RAM, CPU use, and approximate power draw.

Deliverable: one command that produces the tables and plots needed for the final report from a set of recorded trials.

### Phase 5: FedRep Variant

Only add FedRep after the baseline model and fault dictionary work. The implementation should reuse the same encoder/decoder classes rather than duplicating model code.

- Split the SDAE into a shared encoder and local decoder.
- Implement freeze/unfreeze training utilities so encoder-only and decoder-only training are explicit.
- Simulate several clients locally first, using different environmental baselines as different USVs.
- Implement central averaging of encoder weights.
- After encoder averaging, recalculate the global fault dictionary using stored raw fault samples passed through the new encoder.
- Push the averaged encoder and updated dictionary back to each local client.
- Train local decoders on local healthy baseline data while keeping the encoder frozen.

Deliverable: a FedRep training/evaluation workflow that can compare Baseline 0 fault dictionary performance against Baselines 1-4.

### Phase 6: DANN Variant

The DANN implementation should also reuse the baseline SDAE components.

- Train the standard SDAE first.
- Discard or replace the initial decoder as described in the report.
- Add a DANN feature extractor after the SDAE encoder output.
- Add a domain classifier trained with binary/multiclass cross-entropy depending on the number of baselines being used in a run.
- Implement gradient reversal or an equivalent negative domain-loss term with a configurable lambda schedule.
- Train on healthy data from multiple baselines with domain labels.
- Discard the domain classifier after training and deploy only the encoder, DANN feature extractor, decoder path needed for reconstruction, and fault dictionary.
- Track domain classifier BCE/confusion during training as the direct evidence of domain invariance.

Deliverable: a DANN model artifact and evaluation results comparable to the FedRep artifact on the same cross-domain trials.

### Phase 7: Edge Deployment and Profiling

After the models work offline, tighten the Raspberry Pi runtime.

- Export the model with TorchScript first; evaluate ONNX Runtime only if needed.
- Profile end-to-end latency for window construction, inference, and triggered HDBSCAN.
- Measure peak RAM and CPU usage during normal inference and during clustering.
- Quantise or prune only if the model update size or runtime misses the report's SWaP-C targets.
- Keep raw telemetry uploads optional and batch-based; operational sharing should favour model weights, dictionary entries, and selected fault samples.
- Add a watchdog-style runtime wrapper so serial failures, malformed packets, and model errors are logged cleanly rather than silently corrupting a trial.

Deliverable: a Pi-compatible runtime that can run the trained pipeline live and produce logged outputs for the evaluation harness.

### Phase 8: Final Comparison

The final experimental matrix should be built around the report's baseline definitions:

- Baseline 0: nominal operation.
- Baseline 1: power cycling from nominal 16 V to nominal 12 V.
- Baseline 2: bio-fouling via propeller surface modification.
- Baseline 3: thermal shift from approximately 22 C to 30 C.
- Baseline 4: kinematic shock and propeller ventilation using vibration injection and PWM drops/spikes.

The comparison should first establish the fault dictionary using Baseline 0, then evaluate how FedRep and DANN preserve fault identity across Baselines 1-4. Keep the final analysis focused on whether the additional distributed/domain adaptation machinery actually improves cross-domain fault identification enough to justify its complexity.

## 5. Practical Development Order

The first useful coding milestone should be:

```text
raw/synthetic telemetry -> processed windows -> SDAE training -> reconstruction threshold -> replay anomaly detection
```

The second milestone should be:

```text
trained SDAE -> latent rolling buffer -> HDBSCAN -> fault dictionary -> known/novel decision
```

Only after those work should the project branch into:

```text
FedRep comparison
DANN comparison
Raspberry Pi live deployment
final experiment automation
```

This ordering follows the report's critical path: model implementation can progress while hardware is being acquired, but empirical validation depends on the physical test setup. It also avoids prematurely optimising the distributed architectures before the base diagnostic pipeline has proven it can detect and isolate faults at all.

## 6. Main Risks to Manage

- The 2109 -> 420 architecture may be too large or too compressed for the actual data. Keep layer sizes configurable from the start.
- HDBSCAN can be unstable in high-dimensional spaces. If clusters are poor, test lower latent sizes, PCA/UMAP only for analysis, and stronger latent sparsity before changing the whole architecture.
- The reconstruction threshold will likely be baseline-sensitive. Threshold calibration should be logged per model and per baseline.
- The Pi may handle inference but struggle with clustering bursts. Profile triggered clustering early, not at the end.
- Fault labels will be partially manual. Trial metadata discipline is therefore as important as model code.
- FedRep and DANN both depend on consistent artifact versioning. Every model, scaler, dictionary, and dataset manifest needs a version identifier.
- Debugging outputs should be available, such as recontruction loss and clusters logged.

## 7. Definition of Done for the Implementation

The implementation is complete enough for thesis experiments when it can:

- Collect and replay test-rig telemetry with reliable timestamps.
- Train a baseline SDAE on healthy data.
- Detect anomalies using reconstruction error.
- Cluster latent vectors after a fault trigger.
- Identify known faults or declare novel faults using a stored dictionary.
- Run the same evaluation harness for baseline SDAE, FedRep, and DANN variants.
- Produce the report's detection, isolation, drift, and SWaP-C metrics from saved experiment outputs.
- Run live or near-live on the Raspberry Pi 5 without exceeding the practical compute envelope.
