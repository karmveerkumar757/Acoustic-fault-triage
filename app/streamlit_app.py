"""Phase 6 Dashboard: Live Spatial Multi-Mic DSP & Hybrid Unsupervised Fault Triage."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deployment.inference import AcousticTriageEngine
from src.dsp.audio_io import load_stereo_audio
from src.utils.config import load_config

st.set_page_config(
    page_title="AcousticPulse DCASE 2026 — Hybrid Spatial Fault Triage",
    page_icon="🔊",
    layout="wide",
)


@st.cache_resource
def load_engine():
    cfg = load_config("configs/config.yaml")
    return cfg, AcousticTriageEngine(cfg)


def wav_bytes(signal: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    return buf.getvalue()


def main() -> None:
    st.title("AcousticPulse v2: Spatial Multi-Mic Unsupervised Fault Triage")
    st.caption(
        "2-Channel Spatial DSP (Ch0 + Ch1 + Inter-Channel Level Difference) -> Hybrid DS-CNN ONNX -> Mahalanobis + Reconstruction Fusion"
    )

    try:
        cfg, engine = load_engine()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Run the pipeline first:\n"
            "1. `python scripts/train_embeddings.py`\n"
            "2. `python scripts/evaluate_dcase.py`\n"
            "3. `python scripts/export_and_benchmark.py`"
        )
        return

    sr = int(cfg["audio"]["sample_rate"])
    machines = engine.list_machines()

    with st.sidebar:
        st.header("Triage & Spatial DSP Controls")
        machine = st.selectbox("Target Machine Type", options=machines)
        dsp_mode = st.selectbox(
            "Active DSP Front-End",
            options=["spatial_dual_mic", "spatial_dual_mic_gated", "single_mic_gated", "raw_ch0"],
            index=0,
        )
        st.divider()
        st.markdown("**Runtime:** ONNX Runtime (CPU)")
        st.markdown(f"**Calibrated Threshold ({machine}):** `{engine.bank.get_threshold(machine):.3f}`")

    uploaded = st.file_uploader(
        "Upload a stereo (2-channel) DCASE 2026 machine recording (.wav)", type=["wav"]
    )
    if uploaded is None:
        st.info("Upload any `.wav` clip from `data/raw/dcase2026/<machine>/test/` to inspect live triage.")
        return

    audio_2d, _ = load_stereo_audio(io.BytesIO(uploaded.read()), target_sr=sr, window_sec=None)
    res = engine.predict_stereo(audio_2d, machine=machine, dsp_mode=dsp_mode)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Triage Verdict", res["label"])
    c2.metric(
        "Hybrid Anomaly Score",
        f"{res['anomaly_score']:.3f}",
        delta=f"{res['margin_over_threshold']:+.3f} vs thresh",
        delta_color="inverse",
    )
    c3.metric("ONNX + Scorer Latency", f"{res['inference_latency_ms']:.2f} ms")
    c4.metric("Total End-to-End", f"{res['total_latency_ms']:.2f} ms ({res['num_windows']} win)")

    s1, s2 = st.columns(2)
    s1.caption(f"Embedding Mahalanobis/kNN Distance: `{res['embedding_distance']:.4f}`")
    s2.caption(f"Top-Frame Spectrogram Reconstruction MSE: `{res['reconstruction_error']:.5f}`")

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.subheader("Mic 0 (Raw)")
        st.audio(wav_bytes(res["near_waveform"], sr), format="audio/wav")
    with ac2:
        st.subheader("Mic 1 (Raw)")
        if res["far_waveform"] is not None:
            st.audio(wav_bytes(res["far_waveform"], sr), format="audio/wav")
        else:
            st.caption("Mono clip — no second channel.")
    with ac3:
        st.subheader(f"Post-DSP Ch0 ({dsp_mode})")
        st.audio(wav_bytes(res["cleaned_waveform"], sr), format="audio/wav")

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        st.subheader("FFT Spectrum (Ch0 vs Ch1 vs Post-DSP)")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        nf, ndb = res["near_fft"]
        cf, cdb = res["clean_fft"]
        ax.plot(nf, ndb, alpha=0.5, label="Mic 0 Raw", color="#d62728")
        if res["far_fft"] is not None:
            ff, fdb = res["far_fft"]
            ax.plot(ff, fdb, alpha=0.45, label="Mic 1 Raw", color="#7f7f7f")
        ax.plot(cf, cdb, alpha=0.9, label="Post-DSP Ch0", color="#1f77b4")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(0, sr / 2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with vc2:
        st.subheader("128-Bin Log-Mel (Ch0)")
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))
        img = librosa.display.specshow(
            res["log_mel"],
            sr=sr,
            hop_length=int(cfg["dsp"]["hop_length"]),
            x_axis="time",
            y_axis="mel",
            fmin=float(cfg["dsp"]["fmin"]),
            fmax=float(cfg["dsp"]["fmax"]),
            ax=ax2,
            cmap="magma",
        )
        fig2.colorbar(img, ax=ax2, format="%+2.1f")
        st.pyplot(fig2)
        plt.close(fig2)

    with vc3:
        title_ch2 = "Spatial ILD Map (Ch0 - Ch1)" if "spatial" in dsp_mode else "Delta-2 Feature Map"
        st.subheader(title_ch2)
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.6))
        img3 = librosa.display.specshow(
            res["ild_map"],
            sr=sr,
            hop_length=int(cfg["dsp"]["hop_length"]),
            x_axis="time",
            y_axis="mel",
            fmin=float(cfg["dsp"]["fmin"]),
            fmax=float(cfg["dsp"]["fmax"]),
            ax=ax3,
            cmap="coolwarm",
        )
        fig3.colorbar(img3, ax=ax3, format="%+2.1f")
        st.pyplot(fig3)
        plt.close(fig3)


if __name__ == "__main__":
    main()
