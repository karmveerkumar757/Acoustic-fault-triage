"""Digital Signal Processing filters preserving calibrated sensor energy for DCASE 2026."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, istft, sosfiltfilt, stft, wiener


def bandpass_filter(
    signal: np.ndarray,
    sr: int,
    lowcut_hz: float = 50.0,
    highcut_hz: float = 7800.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass filter via second-order sections."""
    nyquist = 0.5 * sr
    low = max(lowcut_hz / nyquist, 1e-4)
    high = min(highcut_hz / nyquist, 0.999)
    if low >= high or len(signal) <= 3 * (2 * order + 1):
        return signal.astype(np.float32)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def _match_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    if len(signal) == target_len:
        return signal
    if len(signal) > target_len:
        return signal[:target_len]
    return np.pad(signal, (0, target_len - len(signal)), mode="constant")


def single_mic_spectral_gate(
    signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    prop_decrease: float = 0.70,
    noise_percentile: float = 15.0,
) -> np.ndarray:
    """Stationary STFT spectral gating that suppresses steady factory floor while preserving transients."""
    noverlap = n_fft - hop_length
    _, _, Z = stft(signal, fs=sr, nperseg=n_fft, noverlap=noverlap)
    mag = np.abs(Z)
    phase = np.angle(Z)

    noise_floor = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    noise_std = np.std(mag, axis=1, keepdims=True)
    thresh = noise_floor + 0.30 * noise_std

    snr_ratio = np.clip((mag - thresh) / (mag + 1e-8), 0.0, 1.0)
    mask = 1.0 - prop_decrease * (1.0 - snr_ratio)

    _, cleaned = istft(mag * mask * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=noverlap)
    return _match_length(cleaned, len(signal)).astype(np.float32)


def dual_channel_spectral_subtraction(
    near_signal: np.ndarray,
    far_signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    alpha: float = 0.65,
    beta: float = 0.15,
    gain_match: bool = True,
) -> np.ndarray:
    """Gentle Dual-Channel Spectral Subtraction (retained for audio playback & diagnostics)."""
    n = min(len(near_signal), len(far_signal))
    noverlap = n_fft - hop_length
    _, _, Z_near = stft(near_signal[:n], fs=sr, nperseg=n_fft, noverlap=noverlap)
    _, _, Z_far = stft(far_signal[:n], fs=sr, nperseg=n_fft, noverlap=noverlap)

    mag_near = np.abs(Z_near)
    phase_near = np.angle(Z_near)
    mag_far = np.abs(Z_far)

    if gain_match:
        floor_near = np.percentile(mag_near, 20.0, axis=1, keepdims=True)
        floor_far = np.percentile(mag_far, 20.0, axis=1, keepdims=True) + 1e-8
        gain = np.clip(floor_near / floor_far, 0.35, 1.5)
    else:
        gain = 1.0

    mag_clean = np.maximum(mag_near - alpha * gain * mag_far, beta * mag_near)
    _, cleaned = istft(mag_clean * np.exp(1j * phase_near), fs=sr, nperseg=n_fft, noverlap=noverlap)
    return _match_length(cleaned, len(near_signal)).astype(np.float32)


def wiener_denoise(signal: np.ndarray, mysize: int = 29) -> np.ndarray:
    """1D adaptive Wiener filter."""
    cleaned = wiener(signal.astype(np.float64), mysize=mysize)
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned.astype(np.float32)


def preprocess_channel(
    signal: np.ndarray,
    sr: int = 16000,
    apply_bandpass: bool = True,
    lowcut_hz: float = 50.0,
    highcut_hz: float = 7800.0,
    filter_order: int = 4,
    spectral_gate: bool = False,
    n_fft: int = 1024,
    hop_length: int = 256,
    prop_decrease: float = 0.70,
    noise_percentile: float = 15.0,
) -> np.ndarray:
    """Clean a single audio channel while PRESERVING true calibrated RMS amplitude."""
    y = signal.astype(np.float32) - float(np.mean(signal))
    if apply_bandpass:
        y = bandpass_filter(y, sr=sr, lowcut_hz=lowcut_hz, highcut_hz=highcut_hz, order=filter_order)
    if spectral_gate:
        y = single_mic_spectral_gate(
            y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            prop_decrease=prop_decrease,
            noise_percentile=noise_percentile,
        )
    return y.astype(np.float32)
