"""Joint training loop: Masked Spectrogram Reconstruction + Sub-Cluster Proxy Learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.ds_cnn import DCaseHybridModel
from src.utils.metrics import plot_training_curves


def _recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.7 * F.mse_loss(recon, target) + 0.3 * F.l1_loss(recon, target)


def evaluate_hybrid(
    model: DCaseHybridModel,
    loader: DataLoader,
    ce_loss_fn: nn.Module,
    recon_weight: float,
    proxy_weight: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    correct = 0
    n = 0

    with torch.no_grad():
        for x_in, x_tgt, y in loader:
            x_in, x_tgt, y = x_in.to(device), x_tgt.to(device), y.to(device)
            _, recon, logits = model(x_in)
            l_rec = _recon_loss(recon, x_tgt)
            l_cls = ce_loss_fn(logits, y)
            loss = recon_weight * l_rec + proxy_weight * l_cls

            bs = x_in.size(0)
            total_loss += float(loss.item()) * bs
            total_recon += float(l_rec.item()) * bs
            correct += int((logits.argmax(dim=1) == y).sum().item())
            n += bs

    return total_loss / max(n, 1), total_recon / max(n, 1), correct / max(n, 1)


def train_hybrid_model(
    model: DCaseHybridModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_to_idx: Dict[str, int],
    cfg: Dict[str, Any],
    device: torch.device,
    save_checkpoint: bool = True,
    checkpoint_override: str | Path | None = None,
) -> Dict[str, Any]:
    """Train DCaseHybridModel strictly on normal train/ clips."""
    tcfg = cfg["training"]
    epochs = int(tcfg["epochs"])
    recon_w = float(tcfg.get("recon_weight", 1.0))
    proxy_w = float(tcfg.get("proxy_weight", 0.25))
    patience = int(tcfg.get("early_stopping_patience", 7))

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    ckpt_path = Path(checkpoint_override or cfg["paths"]["checkpoint_path"])
    report_dir = Path(cfg["paths"]["report_dir"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        n = 0

        for x_in, x_tgt, y in train_loader:
            x_in, x_tgt, y = x_in.to(device), x_tgt.to(device), y.to(device)
            optimizer.zero_grad()

            _, recon, logits = model(x_in)
            l_rec = _recon_loss(recon, x_tgt)
            l_cls = ce_loss_fn(logits, y)
            loss = recon_w * l_rec + proxy_w * l_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = x_in.size(0)
            running_loss += float(loss.item()) * bs
            running_recon += float(l_rec.item()) * bs
            n += bs

        scheduler.step()
        tr_loss = running_loss / max(n, 1)
        tr_rec = running_recon / max(n, 1)
        val_loss, val_rec, val_acc = evaluate_hybrid(
            model, val_loader, ce_loss_fn, recon_w, proxy_w, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_recon": val_rec,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f} (Recon: {tr_rec:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Recon: {val_rec:.4f}) | Val Proxy Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_checkpoint:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "full_state_dict": best_state,
                        "backbone_state_dict": {
                            k.replace("backbone.", "", 1): v
                            for k, v in best_state.items()
                            if k.startswith("backbone.")
                        },
                        "class_to_idx": class_to_idx,
                        "config": cfg,
                        "best_val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_checkpoint:
        plot_training_curves(history, report_dir / "training_curves.png")

    return {"best_val_loss": best_val_loss, "history": history}
