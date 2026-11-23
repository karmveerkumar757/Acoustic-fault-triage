# AcousticPulse: Noise-Aware Unsupervised Acoustic Fault Triage & Edge Pipeline (DCASE 2026)

An end-to-end **unsupervised anomalous sound detection (ASD)** and high-throughput edge inference pipeline built on the **DCASE 2026 Task 2** 2-channel industrial dataset (`bearingEmu`, `fan`, `gearboxEmu`, `valveEmu`).

Combines multi-microphone Digital Signal Processing (**4th-order Butterworth bandpass**, **128-bin calibrated Log-Mel spectrograms**, and **Spatial Inter-Channel Level Difference**) with a lightweight **377K-parameter Depthwise Separable 2D-CNN** trained strictly on normal recordings, deployed via **ONNX Runtime** (**2.01 ms/window**, **4.4x faster** than PyTorch CPU), **FastAPI**, and **Streamlit**.

---

## System Architecture

```text
Stereo Machine Audio (16 kHz, 2-Ch: Mic 0 + Mic 1)
  └── Multi-Mic DSP Front-End (DC Removal -> Butterworth Bandpass 50–7800 Hz)
        └── 3-Channel Spatial Feature Tensor [Log-Mel(Ch0), Log-Mel(Ch1), ILD = Ch0 - Ch1] (3 x 128 x 94)
              └── Hybrid AcousticDSCNN Backbone (376.9K params, 0.175 MB ONNX)
                    ├── Training Only: Masked Spectrogram Reconstruction + Sub-Cluster Cosine Proxy (K=8)
                    └── Edge Inference: L2-Normalized 256-D Embedding -> Prototype-Balanced 1-NN Scorer
                          └── FastAPI Microservice (/predict) & Streamlit Diagnostic Dashboard