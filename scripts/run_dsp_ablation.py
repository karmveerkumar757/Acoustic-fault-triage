"""Phase 5 Scientifically Fair Multi-Mic DSP Ablation Study.

Trains and evaluates a matched model for each DSP front-end so there is ZERO train/test
DSP mismatch:
  1. raw_ch0                (Single-Mic Raw Baseline)
  2. single_mic_gated       (Single-Mic Bandpass + Spectral Gating)
  3. spatial_dual_mic       (2-Mic Spatial: Ch0 + Ch1 + Inter-Channel Level Difference)
  4. spatial_dual_mic_gated (2-Mic Spatial + Spectral Gating)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_dcase import evaluate_dsp_pipeline
from src.data.dcase_dataset import build_proxy_dataloaders
from src.models.ds_cnn import DCaseHybridModel
from src.training.trainer import train_hybrid_model
from src.utils.config import load_config, resolve_device, seed_everything
from src.utils.metrics import plot_dsp_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair matched-mode Multi-Mic DSP Ablation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per ablation mode")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw_ch0", "single_mic_gated", "spatial_dual_mic", "spatial_dual_mic_gated"],
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    device = resolve_device(str(base_cfg["project"]["device"]))
    ablation_ckpts = Path("artifacts/checkpoints/ablation")
    ablation_ckpts.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for mode in args.modes:
        print(f"\n================ Ablation Mode: {mode} ================")
        cfg = copy.deepcopy(base_cfg)
        cfg["dsp"]["mode"] = mode
        cfg["training"]["epochs"] = args.epochs
        seed_everything(int(cfg["project"]["seed"]))

        train_loader, val_loader, class_to_idx = build_proxy_dataloaders(cfg, override_dsp_mode=mode)
        mcfg = cfg["model"]
        model = DCaseHybridModel(
            in_channels=int(mcfg["in_channels"]),
            embedding_dim=int(mcfg["embedding_dim"]),
            num_classes=len(class_to_idx),
            width_mult=float(mcfg["width_mult"]),
            dropout=float(mcfg["dropout"]),
            sub_clusters=int(mcfg.get("sub_clusters", 8)),
            cosine_scale=float(mcfg.get("cosine_scale", 12.0)),
        )
        ckpt_file = ablation_ckpts / f"best_{mode}.pt"
        train_hybrid_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_to_idx=class_to_idx,
            cfg=cfg,
            device=device,
            save_checkpoint=True,
            checkpoint_override=ckpt_file,
        )

        df, _ = evaluate_dsp_pipeline(
            cfg, dsp_mode=mode, checkpoint_override=ckpt_file, save_outputs=False
        )
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    report_dir = Path(base_cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(report_dir / "dcase_dsp_ablation.csv", index=False)
    plot_dsp_ablation(full_df, report_dir / "dcase_dsp_ablation.png")

    print("\n=== Matched-Mode Multi-Mic DSP Ablation Study ===")
    print(full_df.to_string(index=False))
    print(f"\nSaved ablation chart -> {(report_dir / 'dcase_dsp_ablation.png').resolve()}")


if __name__ == "__main__":
    main()
