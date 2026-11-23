# AcousticPulse (DCASE 2026): Noise-Aware Unsupervised Acoustic Fault Triage

An end-to-end **DCASE 2026 Task 2** unsupervised anomalous sound detection & edge deployment pipeline engineered for noisy, 2-channel industrial environments.

## Key Engineering Highlights
* **Multi-Microphone DSP Front-End (`src/dsp/filters.py`)**: Exploits DCASE 2026 stereo recordings (**Channel 0 = Near-Mic**, **Channel 1 = Far-Mic**) via **Dual-Channel Spectral Subtraction** with adaptive per-frequency floor gain matching, benchmarked directly against Raw Near-Mic and Single-Mic Spectral Gating baselines.
* **Strict Unsupervised ASD Protocol (`src/data/dcase_dataset.py`, `src/scoring/anomaly_scorer.py`)**: Trains a compact **Depthwise Separable 2D-CNN with Squeeze-and-Excitation (~280K params)** solely on `train/` normal recordings via a **Cosine-Margin proxy classifier**, then detects unseen faults on `test/` using a **Domain-Generalized Cosine kNN / Ledoit-Wolf Mahalanobis** normal reference bank ($\min(d_{\text{source}}, d_{\text{target}})$).
* **High-Throughput Edge Serving (`src/deployment/`, `app/`)**: Exports the embedding backbone to **ONNX Runtime** with verified numerical parity ($<10^{-4}$ max diff), CPU latency/throughput benchmarking, an interactive **Streamlit** FFT/Spectrogram UI, and a **FastAPI** `/predict` microservice.

---

## Execution Workflow (Step-by-Step)

### 1. Place DCASE 2026 Task 2 Zips (Zenodo `19336329`)
Extract `dev_bearingEmu.zip`, `dev_fan.zip`, `dev_gearboxEmu.zip`, and `dev_valveEmu.zip` into `data/raw/dcase2026/` so each machine folder contains `train/` and `test/`.

*(Or generate a 2-channel synthetic DCASE 2026 dataset for an instant 2-minute smoke test:)*
```bash
python scripts/generate_synthetic_dcase.py
```

### 2. Verify Unit Tests
```bash
pytest -v
```

### 3. Phase 4 — Train Normal-Only Embedding Backbone (GPU Accelerated)
```bash
# On real DCASE 2026 data:
python scripts/train_embeddings.py

# Or on synthetic quickstart data:
python scripts/train_embeddings.py --dataset-root data/raw/synthetic_dcase2026 --epochs 10
```

### 4. Phase 5 — Fit Reference Bank, Official DCASE Metrics & Multi-Mic DSP Ablation
```bash
python scripts/evaluate_dcase.py
python scripts/run_dsp_ablation.py
```

### 5. Phase 6 — Export ONNX Embedder, Benchmark CPU Throughput & Launch Demo Apps
```bash
python scripts/export_and_benchmark.py
streamlit run app/streamlit_app.py
# Optional FastAPI server:
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
