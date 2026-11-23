"""Official DCASE 2026 Task 2 metrics (Source AUC, Target AUC, pAUC@0.1, Harmonic Mean) & plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


def harmonic_mean(values: List[float], eps: float = 1e-8) -> float:
    """Compute official DCASE harmonic mean (Omega), ignoring NaNs."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    arr = np.maximum(arr, eps)
    return float(len(arr) / np.sum(1.0 / arr))


def compute_dcase_machine_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    domains: List[str],
    pauc_max_fpr: float = 0.1,
) -> Dict[str, float]:
    """Compute official DCASE Task 2 metrics for a single machine type.

    Returns:
      - auc_source: ROC-AUC on source-domain normal vs. source-domain anomaly
      - auc_target: ROC-AUC on target-domain normal vs. target-domain anomaly
      - auc_all:    Combined ROC-AUC across both domains
      - pauc:       Combined partial AUC (max_fpr = 0.1)
      - hmean:      Harmonic mean of (auc_source, auc_target, pauc)
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    domains_arr = np.asarray([d.lower() for d in domains])

    def _safe_auc(y_sub: np.ndarray, s_sub: np.ndarray, max_fpr: float | None = None) -> float:
        if len(np.unique(y_sub)) < 2:
            return float("nan")
        if max_fpr is None:
            return float(roc_auc_score(y_sub, s_sub))
        return float(roc_auc_score(y_sub, s_sub, max_fpr=max_fpr))

    src_mask = domains_arr == "source"
    tgt_mask = domains_arr == "target"

    auc_source = _safe_auc(y_true[src_mask], scores[src_mask])
    auc_target = _safe_auc(y_true[tgt_mask], scores[tgt_mask])
    auc_all = _safe_auc(y_true, scores)
    pauc = _safe_auc(y_true, scores, max_fpr=pauc_max_fpr)
    hmean = harmonic_mean([auc_source, auc_target, pauc])

    return {
        "auc_source": auc_source,
        "auc_target": auc_target,
        "auc_all": auc_all,
        "pauc": pauc,
        "hmean": hmean,
    }


def plot_training_curves(history: List[Dict[str, float]], save_path: str | Path) -> None:
    """Plot proxy training loss and validation accuracy over epochs."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.set_style("whitegrid")

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Normal-Only Proxy Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cosine-Margin Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_acc"], label="Val Proxy Acc", color="green", marker="o")
    axes[1].set_title("Validation Machine-Representation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_machine_roc_and_scores(
    machine: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    save_path: str | Path,
) -> None:
    """Save side-by-side ROC curve and Normal vs. Anomaly score distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    sns.set_style("whitegrid")

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_v = roc_auc_score(y_true, scores)
        pauc_v = roc_auc_score(y_true, scores, max_fpr=0.1)
        axes[0].plot(fpr, tpr, lw=2.2, label=f"AUC={auc_v:.4f} | pAUC@0.1={pauc_v:.4f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0].axvspan(0, 0.1, color="orange", alpha=0.12, label="pAUC Zone")
        axes[0].legend(loc="lower right")
    axes[0].set_title(f"{machine} — ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    sns.histplot(scores[y_true == 0], color="#1f77b4", alpha=0.55, kde=True, label="Normal", ax=axes[1])
    sns.histplot(scores[y_true == 1], color="#d62728", alpha=0.55, kde=True, label="Anomaly", ax=axes[1])
    axes[1].axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold ({threshold:.3f})")
    axes[1].set_title(f"{machine} — Unsupervised Anomaly Score Distribution")
    axes[1].set_xlabel("Anomaly Score (Cosine kNN / Mahalanobis Distance)")
    axes[1].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_dsp_ablation(ablation_df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot grouped bar chart comparing DSP front-ends across machines (Harmonic Mean)."""
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    sns.barplot(
        data=ablation_df,
        x="machine",
        y="hmean",
        hue="dsp_mode",
        palette="viridis",
    )
    plt.ylabel("Official DCASE Harmonic Mean (Ω)")
    plt.xlabel("Machine Type")
    plt.title("Multi-Mic DSP Ablation: Raw Near-Mic vs. Single-Mic vs. Dual-Mic Spectral Subtraction")
    plt.ylim(0.4, 1.02)
    plt.legend(title="DSP Front-End")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
