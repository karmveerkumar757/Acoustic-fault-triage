"""Stereo-consistent waveform augmentations and SpecAugment."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def generate_pink_factory_noise(length: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize realistic 1/f pink industrial background noise with 50/100Hz mains hum."""
    white = rng.standard_normal(length)
    freqs = np.fft.rfftfreq(length)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 1.0 / max(length, 1)))
    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=length)
    pink = pink / (np.std(pink) + 1e-8)

    t = np.arange(length) / float(sr)
    hum = 0.12 * np.sin(2.0 * np.pi * 50.0 * t + rng.uniform(0, 2 * np.pi))
    hum += 0.06 * np.sin(2.0 * np.pi * 100.0 * t + rng.uniform(0, 2 * np.pi))
    noise = pink + hum
    return (noise / (np.std(noise) + 1e-8)).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into clean signal at an exact target SNR (dB)."""
    n = len(clean)
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))[:n]
    else:
        noise = noise[:n]
    sig_pow = float(np.mean(clean ** 2)) + 1e-10
    noise_pow = float(np.mean(noise ** 2)) + 1e-10
    target_noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    return (clean + np.sqrt(target_noise_pow / noise_pow) * noise).astype(np.float32)


def apply_spec_augment(
    spec: np.ndarray,
    freq_mask_param: int = 12,
    time_mask_param: int = 12,
    num_masks: int = 1,
) -> np.ndarray:
    """Apply light frequency & time masking on (C, F, T) spectrograms."""
    out = spec.copy()
    _, n_freq, n_time = out.shape
    for _ in range(num_masks):
        fw = random.randint(0, min(freq_mask_param, n_freq - 1))
        if fw > 0:
            f0 = random.randint(0, n_freq - fw)
            out[:, f0 : f0 + fw, :] = 0.0
        tw = random.randint(0, min(time_mask_param, n_time - 1))
        if tw > 0:
            t0 = random.randint(0, n_time - tw)
            out[:, :, t0 : t0 + tw] = 0.0
    return out


class StereoWaveformAugmenter:
    """Applies phase-synchronized mild gain & time shifts to (C, N) stereo waveforms."""

    def __init__(
        self,
        sr: int = 16000,
        gain_db_range: Tuple[float, float] = (-3.0, 3.0),
        time_shift_prob: float = 0.5,
    ):
        self.sr = sr
        self.gain_db_range = gain_db_range
        self.time_shift_prob = time_shift_prob

    def __call__(self, audio_2d: np.ndarray) -> np.ndarray:
        y = audio_2d.astype(np.float32).copy()
        if random.random() < self.time_shift_prob:
            shift = int(random.uniform(-0.2, 0.2) * y.shape[-1])
            y = np.roll(y, shift, axis=-1)
        gain_db = random.uniform(self.gain_db_range[0], self.gain_db_range[1])
        y = y * (10.0 ** (gain_db / 20.0))
        return y.astype(np.float32)
