"""Generate a realistic 2-channel stereo DCASE 2026 Task 2 synthetic dataset.

Creates 4 machines (bearingEmu, fan, gearboxEmu, valveEmu) with the EXACT official
DCASE 2026 layout:
  - train/: Normal clips ONLY (source + target domains)
  - test/:  Normal + Anomaly clips (source + target domains)
  - 2-channel 16kHz WAV: Channel 0 = Near-mic (machine + ambient), Channel 1 = Far-mic (mostly ambient)

Use this to smoke-test the full unsupervised + dual-mic pipeline in ~2 minutes before
or while downloading Zenodo 19336329.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dsp.augmentations import generate_pink_factory_noise


def _machine_signal(machine: str, domain: str, is_anomaly: bool, sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n) / float(sr)
    shift = 1.15 if domain == "target" else 1.0  # Domain shift (speed/load variation)

    if machine == "bearingEmu":
        f0 = rng.uniform(135.0, 155.0) * shift
        y = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t) + 0.12 * rng.normal(0, 1, n)
        if is_anomaly:
            # Periodic outer-race bearing impact train at ~14-20 Hz
            step = int(sr / rng.uniform(14.0, 20.0))
            imp_len = int(0.015 * sr)
            it = np.arange(imp_len) / sr
            ring = np.exp(-280.0 * it) * np.sin(2 * np.pi * rng.uniform(2800, 4600) * it)
            for p in range(rng.integers(0, step), n - imp_len, step):
                y[p : p + imp_len] += rng.uniform(0.7, 1.1) * ring

    elif machine == "fan":
        f0 = rng.uniform(85.0, 105.0) * shift
        # Smooth aerodynamic broadband + blade pass harmonics
        y = 0.45 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
        y += 0.20 * generate_pink_factory_noise(n, sr, rng)
        if is_anomaly:
            # Blade imbalance flutter + high-freq bearing whine
            whine_f = rng.uniform(3200, 5200)
            y += 0.45 * (1.0 + 0.5 * np.sin(2 * np.pi * 6.0 * t)) * np.sin(2 * np.pi * whine_f * t)

    elif machine == "gearboxEmu":
        f0 = rng.uniform(220.0, 260.0) * shift
        y = 0.45 * np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(2 * np.pi * 2.5 * f0 * t) + 0.15 * np.sin(2 * np.pi * 5 * f0 * t)
        if is_anomaly:
            # Chipped tooth amplitude-modulated sideband bursts
            env = (0.5 * (1.0 + np.sin(2 * np.pi * rng.uniform(12.0, 24.0) * t))) ** 3
            y += 0.65 * env * np.sin(2 * np.pi * rng.uniform(1800, 3400) * t)

    else:  # valveEmu
        y = 0.10 * rng.normal(0, 1, n)
        # Normal rhythmic open/close valve clicks every ~1.2s
        period = int(sr * (1.1 if domain == "source" else 0.85))
        click_len = int(0.025 * sr)
        ct = np.arange(click_len) / sr
        click = np.exp(-160.0 * ct) * np.sin(2 * np.pi * 950.0 * ct)
        for p in range(int(0.2 * sr), n - click_len, period):
            y[p : p + click_len] += 0.85 * click
        if is_anomaly:
            # Irregular double-chatter & high-freq pressure leak hiss
            y += 0.35 * np.sin(2 * np.pi * rng.uniform(4500, 6500) * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
            for p in range(int(0.35 * sr), n - click_len, period):
                y[p : p + click_len] += 0.7 * np.exp(-120.0 * ct) * np.sin(2 * np.pi * 2400.0 * ct)

    y = y / (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


def _make_stereo_pair(machine_sig: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Create DCASE 2026 stereo (N, 2): ch0=Near-Mic (strong signal + loud noise), ch1=Far-Mic (weak signal + loud noise)."""
    n = len(machine_sig)
    ambient = generate_pink_factory_noise(n, sr, rng)
    # Add non-stationary factory clanks to ambient noise so Dual-Mic subtraction shines!
    t = np.arange(n) / float(sr)
    ambient = ambient + 0.55 * np.sin(2 * np.pi * rng.uniform(400, 1200) * t) * (np.sin(2 * np.pi * 0.7 * t) > 0.2)
    ambient = ambient / (np.std(ambient) + 1e-8)

    # Near mic sees target machine at ~0 to +4 dB SNR relative to factory floor
    noise_gain = float(rng.uniform(0.55, 1.15))
    near = machine_sig + noise_gain * ambient
    # Far mic hears attenuated machine (12-20%) + slightly scaled ambient noise
    far_mic_scale = float(rng.uniform(0.85, 1.25))
    far = 0.15 * machine_sig + (noise_gain * far_mic_scale) * ambient + 0.03 * rng.normal(0, 1, n)

    stereo = np.stack([near, far], axis=1).astype(np.float32)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    return (stereo / peak) * 0.9


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2-channel synthetic DCASE 2026 Task 2 dataset.")
    parser.add_argument("--out-dir", type=str, default="data/raw/synthetic_dcase2026")
    parser.add_argument("--train-source", type=int, default=80)
    parser.add_argument("--train-target", type=int, default=10)
    parser.add_argument("--test-per-group", type=int, default=20, help="Clips per (domain, label) in test/")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    machines = ["bearingEmu", "fan", "gearboxEmu", "valveEmu"]
    root = Path(args.out_dir)

    for m in machines:
        tr_dir = root / m / "train"
        te_dir = root / m / "test"
        tr_dir.mkdir(parents=True, exist_ok=True)
        te_dir.mkdir(parents=True, exist_ok=True)

        # Normal-only train/
        for i in range(args.train_source):
            sig = _machine_signal(m, "source", False, args.sr, args.duration, rng)
            sf.write(tr_dir / f"section_00_source_train_normal_{i:04d}.wav", _make_stereo_pair(sig, args.sr, rng), args.sr)
        for i in range(args.train_target):
            sig = _machine_signal(m, "target", False, args.sr, args.duration, rng)
            sf.write(tr_dir / f"section_00_target_train_normal_{i:04d}.wav", _make_stereo_pair(sig, args.sr, rng), args.sr)

        # Test/ with source+target and normal+anomaly
        idx = 0
        for domain in ("source", "target"):
            for is_anom, tag in [(False, "normal"), (True, "anomaly")]:
                for _ in range(args.test_per_group):
                    sig = _machine_signal(m, domain, is_anom, args.sr, args.duration, rng)
                    fname = f"section_00_{domain}_test_{tag}_{idx:04d}.wav"
                    sf.write(te_dir / fname, _make_stereo_pair(sig, args.sr, rng), args.sr)
                    idx += 1

    print(f"Generated 2-channel synthetic DCASE 2026 dataset for {machines} in {root.resolve()}.")


if __name__ == "__main__":
    main()
