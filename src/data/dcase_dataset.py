"""Official DCASE 2026 Task 2 dataset loader with Fine-Grained Attribute Proxy Parsing.

Parses operating attributes embedded in DCASE filenames (e.g., `spd_1`, `pro_A`, `noAttribute`)
so proxy supervision distinguishes operating states instead of collapsing to 4 coarse classes.
Strictly uses ONLY `<machine>/train/*.wav` (100% normal clips) during training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.dsp.audio_io import load_stereo_audio
from src.dsp.augmentations import StereoWaveformAugmenter, apply_spec_augment
from src.dsp.features import StereoAcousticExtractor

_ATTR_RE = re.compile(r"_(?:normal|anomaly|abnormal)_\d+_(.+)$", re.IGNORECASE)


@dataclass
class DCaseClip:
    path: Path
    machine: str
    split: str        # "train" or "test"
    domain: str       # "source" or "target"
    attribute: str    # e.g., "spd_1", "pro_A", "noAttribute"
    label: int        # 0 = normal, 1 = anomaly
    proxy_class: int  # integer ID for fine-grained (machine + attribute) normal proxy


def _parse_metadata(stem: str) -> Tuple[str, int, str]:
    lower = stem.lower()
    domain = "target" if "target" in lower else "source"
    label = 1 if ("anomaly" in lower or "abnormal" in lower) else 0
    m = _ATTR_RE.search(stem)
    attr = m.group(1) if m else "noAttribute"
    return domain, label, attr


def discover_dcase_machines(dataset_root: str | Path) -> List[Path]:
    """Find all machine directories containing both `train/` and `test/` folders."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root.resolve()}.\n"
            "Extract DCASE 2026 zips into data/raw/dcase2026/ or run "
            "`python scripts/generate_synthetic_dcase.py`."
        )

    machines: List[Path] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            machines.append(candidate)
        else:
            for sub in sorted(candidate.iterdir()):
                if sub.is_dir() and (sub / "train").is_dir() and (sub / "test").is_dir():
                    machines.append(sub)

    if not machines:
        raise RuntimeError(f"No machine folders with train/ and test/ found under {root.resolve()}.")
    return machines


def load_dcase_manifest(dataset_root: str | Path) -> Tuple[List[DCaseClip], List[DCaseClip], Dict[str, int]]:
    """Build strict normal-only train/ and full test/ manifests with fine-grained attribute classes."""
    machine_dirs = discover_dcase_machines(dataset_root)

    # First pass: count attribute occurrences to collapse ultra-rare singleton attributes safely
    raw_train = []
    attr_counts: Dict[str, int] = {}
    for mdir in machine_dirs:
        mname = mdir.name
        for wav in sorted((mdir / "train").glob("*.wav")):
            domain, label, attr = _parse_metadata(wav.stem)
            if label != 0:
                continue
            key = f"{mname}__{attr}"
            attr_counts[key] = attr_counts.get(key, 0) + 1
            raw_train.append((wav, mname, domain, attr, key))

    class_to_idx: Dict[str, int] = {}
    train_clips: List[DCaseClip] = []
    for wav, mname, domain, attr, key in raw_train:
        # Merge classes with <5 samples into machine fallback so stratified split never fails
        eff_key = key if attr_counts[key] >= 5 else f"{mname}__other"
        if eff_key not in class_to_idx:
            class_to_idx[eff_key] = len(class_to_idx)
        train_clips.append(
            DCaseClip(
                path=wav,
                machine=mname,
                split="train",
                domain=domain,
                attribute=attr,
                label=0,
                proxy_class=class_to_idx[eff_key],
            )
        )

    test_clips: List[DCaseClip] = []
    for mdir in machine_dirs:
        mname = mdir.name
        for wav in sorted((mdir / "test").glob("*.wav")):
            domain, label, attr = _parse_metadata(wav.stem)
            key = f"{mname}__{attr}"
            proxy_idx = class_to_idx.get(key, 0)
            test_clips.append(
                DCaseClip(
                    path=wav,
                    machine=mname,
                    split="test",
                    domain=domain,
                    attribute=attr,
                    label=label,
                    proxy_class=proxy_idx,
                )
            )

    if not train_clips:
        raise RuntimeError(f"No training .wav files found in {dataset_root}")
    return train_clips, test_clips, class_to_idx


class DCaseHybridTrainDataset(Dataset):
    """Returns (masked_input_spec, clean_target_spec, proxy_class) for Masked AE + SubCluster training."""

    def __init__(
        self,
        clips: List[DCaseClip],
        extractor: StereoAcousticExtractor,
        augmenter: Optional[StereoWaveformAugmenter] = None,
        spec_augment_cfg: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
    ):
        self.clips = clips
        self.extractor = extractor
        self.augmenter = augmenter if is_train else None
        self.spec_augment_cfg = spec_augment_cfg if is_train else None
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip = self.clips[idx]
        audio_2d, _ = load_stereo_audio(
            clip.path,
            target_sr=self.extractor.sr,
            window_sec=self.extractor.window_sec,
            random_crop=self.is_train,
        )

        if self.augmenter is not None:
            audio_2d = self.augmenter(audio_2d)

        clean_spec = self.extractor.extract_single_window(audio_2d)
        input_spec = clean_spec

        # Denoising / Masked Autoencoder trick: mask input spectrogram, reconstruct unmasked target!
        if self.spec_augment_cfg and self.spec_augment_cfg.get("enabled", False):
            input_spec = apply_spec_augment(
                clean_spec,
                freq_mask_param=int(self.spec_augment_cfg.get("freq_mask_param", 12)),
                time_mask_param=int(self.spec_augment_cfg.get("time_mask_param", 12)),
                num_masks=int(self.spec_augment_cfg.get("num_masks", 1)),
            )

        return (
            torch.from_numpy(input_spec),
            torch.from_numpy(clean_spec),
            torch.tensor(clip.proxy_class, dtype=torch.long),
        )


def build_proxy_dataloaders(
    cfg: Dict[str, Any],
    override_dsp_mode: str | None = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build stratified normal-only train/val DataLoaders."""
    train_clips, _, class_to_idx = load_dcase_manifest(cfg["paths"]["dataset_root"])
    labels = [c.proxy_class for c in train_clips]

    tr_clips, val_clips = train_test_split(
        train_clips,
        test_size=float(cfg["training"]["val_split"]),
        stratify=labels,
        random_state=int(cfg["project"]["seed"]),
    )

    extractor = StereoAcousticExtractor(cfg, override_dsp_mode=override_dsp_mode)
    aug_cfg = cfg["augmentation"]
    augmenter = None
    if aug_cfg.get("enabled", True):
        augmenter = StereoWaveformAugmenter(
            sr=int(cfg["audio"]["sample_rate"]),
            gain_db_range=tuple(aug_cfg.get("gain_db_range", [-3.0, 3.0])),
            time_shift_prob=float(aug_cfg.get("time_shift_prob", 0.5)),
        )

    train_ds = DCaseHybridTrainDataset(
        tr_clips,
        extractor=extractor,
        augmenter=augmenter,
        spec_augment_cfg=aug_cfg.get("spec_augment"),
        is_train=True,
    )
    val_ds = DCaseHybridTrainDataset(val_clips, extractor=extractor, is_train=False)

    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, class_to_idx
