"""Stereo-preserving 2-channel audio loader optimized for DCASE 2026 Task 2."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def _resample_channels(audio_2d: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample (C, N) float32 array when file sample rate differs from target_sr."""
    if orig_sr == target_sr:
        return audio_2d.astype(np.float32)
    channels = [
        librosa.resample(audio_2d[c], orig_sr=orig_sr, target_sr=target_sr)
        for c in range(audio_2d.shape[0])
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def load_stereo_audio(
    source: Union[str, Path, io.BytesIO],
    target_sr: int = 16000,
    window_sec: float | None = None,
    random_crop: bool = False,
) -> Tuple[np.ndarray, int]:
    """Load audio while preserving channels as shape (C, N) — NEVER mixes near & far mics.

    If `window_sec` is specified and `source` is a file path, uses fast SoundFile frame
    seeking to read only the required slice instead of decoding the full 10-12s clip.
    """
    if isinstance(source, (str, Path)) and window_sec is not None:
        info = sf.info(str(source))
        orig_sr = int(info.samplerate)
        total_frames = int(info.frames)
        req_frames = int(round(window_sec * orig_sr))

        if total_frames > req_frames:
            if random_crop:
                start = random.randint(0, total_frames - req_frames)
            else:
                start = (total_frames - req_frames) // 2
            data, sr = sf.read(str(source), start=start, frames=req_frames, dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(str(source), dtype="float32", always_2d=True)
    else:
        data, sr = sf.read(source, dtype="float32", always_2d=True)

    # soundfile returns (N, C) -> transpose to (C, N)
    audio = data.T.astype(np.float32)
    audio = _resample_channels(audio, orig_sr=int(sr), target_sr=target_sr)
    return audio, target_sr


def split_near_far(
    audio: np.ndarray,
    near_channel: int = 0,
    far_channel: int = 1,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Split (C, N) array into (near_waveform, far_waveform_or_None)."""
    if audio.ndim == 1:
        return audio.astype(np.float32), None
    num_ch = audio.shape[0]
    near = audio[min(near_channel, num_ch - 1)].astype(np.float32)
    if num_ch >= 2 and far_channel < num_ch and far_channel != near_channel:
        far = audio[far_channel].astype(np.float32)
        return near, far
    return near, None
