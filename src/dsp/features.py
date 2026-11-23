"""High-resolution 128-bin Log-Mel & Spatial Multi-Mic (Ch0, Ch1, ILD) Feature Extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from src.dsp.audio_io import split_near_far
from src.dsp.filters import preprocess_channel


def pad_or_trim_1d(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Center-crop or zero-pad 1D waveform to exact target_length."""
    n = len(signal)
    if n == target_length:
        return signal.astype(np.float32)
    if n > target_length:
        start = (n - target_length) // 2
        return signal[start : start + target_length].astype(np.float32)
    pad_total = target_length - n
    pad_left = pad_total // 2
    return np.pad(signal, (pad_left, pad_total - pad_left), mode="constant").astype(np.float32)


def compute_fft_spectrum(signal: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided Hann-windowed FFT magnitude spectrum (Hz, dB)."""
    n = len(signal)
    if n == 0:
        return np.array([0.0]), np.array([-120.0])
    window = np.hanning(n)
    fft_vals = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(fft_vals) / (n / 2.0 + 1e-8)
    mag_db = 20.0 * np.log10(mag + 1e-8)
    return freqs, mag_db


def compute_calibrated_log_mel(
    signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 50.0,
    fmax: float = 7800.0,
    db_offset: float = 35.0,
    db_scale: float = 30.0,
) -> np.ndarray:
    """Extract calibrated Log-Mel spectrogram using fixed ref=1.0 (preserves absolute energy)."""
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=min(fmax, sr / 2.0),
        power=2.0,
    )
    # ref=1.0 preserves true sensor loudness/energy shifts instead of zeroing out peak amplitude
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
    scaled = (log_mel + db_offset) / max(db_scale, 1e-6)
    return np.clip(scaled, -4.0, 4.0).astype(np.float32)


def _single_mic_3ch(log_mel: np.ndarray) -> np.ndarray:
    """Stack [Log-Mel, Delta, Delta-Delta] for 3-channel single-mic parity."""
    d1 = librosa.feature.delta(log_mel)
    d2 = librosa.feature.delta(log_mel, order=2)
    return np.stack([log_mel, d1 * 2.0, d2 * 2.0], axis=0).astype(np.float32)


def _spatial_dual_mic_3ch(log_mel_0: np.ndarray, log_mel_1: np.ndarray | None) -> np.ndarray:
    """Stack [Log-Mel(Ch0), Log-Mel(Ch1), Inter-Channel Level Difference (ILD = Ch0 - Ch1)].

    ILD is a non-destructive spatial DSP cue: localized machine energy has a consistent
    per-frequency level ratio between Mic 0 and Mic 1, whereas diffuse factory noise does not.
    """
    if log_mel_1 is None:
        d1 = librosa.feature.delta(log_mel_0)
        return np.stack([log_mel_0, log_mel_0, d1 * 2.0], axis=0).astype(np.float32)
    ild = (log_mel_0 - log_mel_1) * 3.0  # Scale ILD (~0.1-0.5 range) to match Log-Mel variance
    return np.stack([log_mel_0, log_mel_1, ild], axis=0).astype(np.float32)


class StereoAcousticExtractor:
    """Config-driven 3-channel Spatial Multi-Mic & Single-Mic DSP Feature Extractor."""

    def __init__(self, cfg: Dict[str, Any], override_dsp_mode: str | None = None):
        audio_cfg = cfg["audio"]
        dsp_cfg = cfg["dsp"]

        self.sr = int(audio_cfg["sample_rate"])
        self.window_sec = float(audio_cfg["window_sec"])
        self.eval_hop_sec = float(audio_cfg.get("eval_hop_sec", 1.75))
        self.window_samples = int(round(self.sr * self.window_sec))
        self.hop_samples = max(int(round(self.sr * self.eval_hop_sec)), 1)
        self.near_ch = int(audio_cfg.get("near_channel", 0))
        self.far_ch = int(audio_cfg.get("far_channel", 1))

        self.mode = (override_dsp_mode if override_dsp_mode is not None else str(dsp_cfg["mode"])).lower()
        self.apply_bandpass = bool(dsp_cfg.get("apply_bandpass", True))
        self.lowcut_hz = float(dsp_cfg.get("lowcut_hz", 50.0))
        self.highcut_hz = float(dsp_cfg.get("highcut_hz", 7800.0))
        self.filter_order = int(dsp_cfg.get("filter_order", 4))
        self.n_fft = int(dsp_cfg["n_fft"])
        self.hop_length = int(dsp_cfg["hop_length"])
        self.n_mels = int(dsp_cfg["n_mels"])
        self.fmin = float(dsp_cfg["fmin"])
        self.fmax = float(dsp_cfg["fmax"])
        self.db_offset = float(dsp_cfg.get("db_offset", 35.0))
        self.db_scale = float(dsp_cfg.get("db_scale", 30.0))

        single_cfg = dsp_cfg.get("single_mic", {})
        self.prop_decrease = float(single_cfg.get("prop_decrease", 0.70))
        self.noise_percentile = float(single_cfg.get("noise_percentile", 15.0))

    def _preprocess_pair(self, audio_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        """Return (proc_ch0, proc_ch1_or_None, raw_ch0, raw_ch1_or_None)."""
        raw_0, raw_1 = split_near_far(audio_2d, near_channel=self.near_ch, far_channel=self.far_ch)

        if self.mode == "raw_ch0":
            proc_0 = raw_0 - float(np.mean(raw_0))
            return proc_0.astype(np.float32), None, raw_0, raw_1

        use_gate = self.mode in ("single_mic_gated", "spatial_dual_mic_gated")
        proc_0 = preprocess_channel(
            raw_0,
            sr=self.sr,
            apply_bandpass=self.apply_bandpass,
            lowcut_hz=self.lowcut_hz,
            highcut_hz=self.highcut_hz,
            filter_order=self.filter_order,
            spectral_gate=use_gate,
            n_fft=self.n_fft,
            hop_length=256,
            prop_decrease=self.prop_decrease,
            noise_percentile=self.noise_percentile,
        )

        proc_1 = None
        if self.mode in ("spatial_dual_mic", "spatial_dual_mic_gated") and raw_1 is not None:
            proc_1 = preprocess_channel(
                raw_1,
                sr=self.sr,
                apply_bandpass=self.apply_bandpass,
                lowcut_hz=self.lowcut_hz,
                highcut_hz=self.highcut_hz,
                filter_order=self.filter_order,
                spectral_gate=use_gate,
                n_fft=self.n_fft,
                hop_length=256,
                prop_decrease=self.prop_decrease,
                noise_percentile=self.noise_percentile,
            )

        return proc_0, proc_1, raw_0, raw_1

    def _window_to_tensor(self, w0: np.ndarray, w1: np.ndarray | None) -> np.ndarray:
        lm0 = compute_calibrated_log_mel(
            w0,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            db_offset=self.db_offset,
            db_scale=self.db_scale,
        )
        if self.mode in ("spatial_dual_mic", "spatial_dual_mic_gated"):
            lm1 = (
                compute_calibrated_log_mel(
                    w1,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    db_offset=self.db_offset,
                    db_scale=self.db_scale,
                )
                if w1 is not None
                else None
            )
            return _spatial_dual_mic_3ch(lm0, lm1)
        return _single_mic_3ch(lm0)

    def extract_single_window(self, audio_2d: np.ndarray) -> np.ndarray:
        """Process a single crop -> (3, n_mels, T)."""
        p0, p1, _, _ = self._preprocess_pair(audio_2d)
        w0 = pad_or_trim_1d(p0, self.window_samples)
        w1 = pad_or_trim_1d(p1, self.window_samples) if p1 is not None else None
        return self._window_to_tensor(w0, w1)

    def extract_sliding_windows(
        self, audio_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Preprocess full 10s clip once and slice overlapping windows -> (W, 3, n_mels, T)."""
        p0, p1, raw_0, raw_1 = self._preprocess_pair(audio_2d)
        n = len(p0)

        windows: List[np.ndarray] = []
        if n <= self.window_samples:
            w0 = pad_or_trim_1d(p0, self.window_samples)
            w1 = pad_or_trim_1d(p1, self.window_samples) if p1 is not None else None
            windows.append(self._window_to_tensor(w0, w1))
        else:
            starts = list(range(0, n - self.window_samples + 1, self.hop_samples))
            if starts[-1] != n - self.window_samples:
                starts.append(n - self.window_samples)
            for s in starts:
                w0 = p0[s : s + self.window_samples]
                w1 = p1[s : s + self.window_samples] if p1 is not None else None
                windows.append(self._window_to_tensor(w0, w1))

        return np.stack(windows, axis=0).astype(np.float32), p0, raw_0, raw_1
