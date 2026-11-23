"""Phase 4 (Iteration 2): Train Hybrid Spatial DS-CNN (Masked AE + Sub-Cluster Proxy) on normal train/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dcase_dataset import build_proxy_dataloaders
from src.models.ds_cnn import DCaseHybridModel, count_parameters
from src.training.trainer import train_hybrid_model
from src.utils.config import load_config, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCASE 2026 Hybrid Spatial DS-CNN.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dsp-mode", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg["paths"]["dataset_root"] = args.dataset_root
    if args.dsp_mode:
        cfg["dsp"]["mode"] = args.dsp_mode
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    seed_everything(int(cfg["project"]["seed"]))
    device = resolve_device(str(cfg["project"]["device"]))

    print(f"Device: {device} | DSP Mode: {cfg['dsp']['mode']} | Dataset: {cfg['paths']['dataset_root']}")
    train_loader, val_loader, class_to_idx = build_proxy_dataloaders(cfg)
    print(
        f"Fine-grained normal proxy classes ({len(class_to_idx)}): {list(class_to_idx.keys())}\n"
        f"Train clips: {len(train_loader.dataset)} | Val clips: {len(val_loader.dataset)}"
    )

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
    print(
        f"Hybrid Backbone params: {count_parameters(model.backbone):,} "
        f"(total w/ sub-cluster head: {count_parameters(model):,})"
    )

    out = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        cfg=cfg,
        device=device,
        save_checkpoint=True,
    )
    print(f"\nTraining complete! Best Val Loss: {out['best_val_loss']:.4f}")
    print(f"Saved checkpoint -> {Path(cfg['paths']['checkpoint_path']).resolve()}")


if __name__ == "__main__":
    main()
