"""Production ONNX Runtime + Hybrid Unsupervised Scorer Inference Engine."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import onnxruntime as ort

from src.dsp.audio_io import load_stereo_audio
from src.dsp.features import StereoAcousticExtractor, compute_fft_spectrum
from src.scoring.anomaly_scorer import MachineReferenceBank, summarize_frame_recon_error


class AcousticTriageEngine:
    """End-to-end edge triage: Stereo Audio -> Spatial DSP -> ONNX Hybrid -> Anomaly Score."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        onnx_path: Optional[str | Path] = None,
        bank_path: Optional[str | Path] = None,
    ):
        self.cfg = cfg
        self.sr = int(cfg["audio"]["sample_rate"])
        self.top_frame_ratio = float(cfg["scoring"].get("recon_top_frame_ratio", 0.15))

        model_path = Path(onnx_path or cfg["paths"]["onnx_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path.resolve()}. "
                "Run `python scripts/export_and_benchmark.py` first."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.bank = MachineReferenceBank.load(bank_path or cfg["paths"]["reference_bank_path"])

    def list_machines(self):
        return self.bank.list_machines()

    def predict_stereo(
        self,
        audio_2d: np.ndarray,
        machine: str,
        dsp_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        extractor = StereoAcousticExtractor(self.cfg, override_dsp_mode=dsp_mode)

        t0 = time.perf_counter()
        windows, proc_ch0, raw_ch0, raw_ch1 = extractor.extract_sliding_windows(audio_2d)
        t_dsp_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        win_embs, win_frame_mse = self.session.run(["embedding", "frame_mse"], {"spectrogram": windows})

        clip_emb = np.mean(win_embs, axis=0, keepdims=True)
        clip_emb = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-10)
        clip_recon = np.array([summarize_frame_recon_error(win_frame_mse, top_ratio=self.top_frame_ratio)])

        hybrid, emb_dist, rec_err = self.bank.score_components(machine, clip_emb, clip_recon)
        anomaly_score = float(hybrid[0])
        t_inf_ms = (time.perf_counter() - t1) * 1000.0

        threshold = self.bank.get_threshold(machine)
        is_anomaly = anomaly_score >= threshold
        margin = anomaly_score - threshold

        near_fft = compute_fft_spectrum(raw_ch0, self.sr)
        clean_fft = compute_fft_spectrum(proc_ch0, self.sr)
        far_fft = compute_fft_spectrum(raw_ch1, self.sr) if raw_ch1 is not None else None

        return {
            "machine": machine,
            "label": "ANOMALY / FAULT" if is_anomaly else "NORMAL",
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(threshold, 4),
            "margin_over_threshold": round(margin, 4),
            "embedding_distance": round(float(emb_dist[0]), 4),
            "reconstruction_error": round(float(rec_err[0]), 5),
            "num_windows": int(windows.shape[0]),
            "dsp_mode": extractor.mode,
            "is_stereo": raw_ch1 is not None,
            "dsp_latency_ms": round(t_dsp_ms, 2),
            "inference_latency_ms": round(t_inf_ms, 2),
            "total_latency_ms": round(t_dsp_ms + t_inf_ms, 2),
            "near_waveform": raw_ch0,
            "far_waveform": raw_ch1,
            "cleaned_waveform": proc_ch0,
            "log_mel": windows[0, 0],
            "ild_map": windows[0, 2],
            "near_fft": near_fft,
            "far_fft": far_fft,
            "clean_fft": clean_fft,
        }

    def predict_source(
        self,
        source: Union[str, Path, io.BytesIO],
        machine: str,
        dsp_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        audio_2d, _ = load_stereo_audio(source, target_sr=self.sr, window_sec=None)
        return self.predict_stereo(audio_2d, machine=machine, dsp_mode=dsp_mode)
