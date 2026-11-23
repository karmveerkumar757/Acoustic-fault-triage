"""Phase 5 (Iteration 2): Fit Hybrid Normal Reference Bank & compute official DCASE 2026 metrics.

Prints per-machine breakdown of:
  - Embedding-Only Harmonic Mean (Mahalanobis / kNN)
  - Reconstruction-Only Harmonic Mean (Top-15% Frame MSE)
  - Hybrid Fusion Harmonic Mean (Official Source AUC, Target AUC, pAUC@0.1, Ω)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dcase_dataset import DCaseClip, load_dcase_manifest
from src.deployment.export_onnx import load_backbone_from_checkpoint
from src.dsp.audio_io import load_stereo_audio
from src.dsp.features import StereoAcousticExtractor
from src.scoring.anomaly_scorer import MachineReferenceBank, summarize_frame_recon_error
from src.utils.config import load_config, resolve_device, seed_everything
from src.utils.metrics import (
    compute_dcase_machine_metrics,
    harmonic_mean,
    plot_machine_roc_and_scores,
)


@torch.no_grad()
def extract_clip_features(
    clips: List[DCaseClip],
    backbone: torch.nn.Module,
    extractor: StereoAcousticExtractor,
    device: torch.device,
    top_frame_ratio: float = 0.15,
    desc: str = "Extracting",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (clip_embeddings [N, D], clip_recon_errors [N,])."""
    embs: List[np.ndarray] = []
    recs: List[float] = []

    for clip in tqdm(clips, desc=desc, leave=False):
        audio_2d, _ = load_stereo_audio(clip.path, target_sr=extractor.sr, window_sec=None)
        windows, _, _, _ = extractor.extract_sliding_windows(audio_2d)
        batch = torch.from_numpy(windows).to(device)
        w_embs, w_frame_mse = backbone(batch)

        w_embs_np = w_embs.cpu().numpy()
        mean_emb = np.mean(w_embs_np, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
        embs.append(mean_emb.astype(np.float32))

        rec_err = summarize_frame_recon_error(w_frame_mse.cpu().numpy(), top_ratio=top_frame_ratio)
        recs.append(rec_err)

    return np.stack(embs, axis=0), np.array(recs, dtype=np.float64)


def evaluate_dsp_pipeline(
    cfg: Dict[str, Any],
    dsp_mode: str | None = None,
    checkpoint_override: str | Path | None = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, MachineReferenceBank]:
    device = resolve_device(str(cfg["project"]["device"]))
    ckpt_path = Path(checkpoint_override or cfg["paths"]["checkpoint_path"])
    backbone = load_backbone_from_checkpoint(ckpt_path, cfg).to(device)
    backbone.eval()

    extractor = StereoAcousticExtractor(cfg, override_dsp_mode=dsp_mode)
    train_clips, test_clips, _ = load_dcase_manifest(cfg["paths"]["dataset_root"])

    train_by_machine: Dict[str, List[DCaseClip]] = defaultdict(list)
    test_by_machine: Dict[str, List[DCaseClip]] = defaultdict(list)
    for c in train_clips:
        train_by_machine[c.machine].append(c)
    for c in test_clips:
        test_by_machine[c.machine].append(c)

    scfg = cfg["scoring"]
    top_ratio = float(scfg.get("recon_top_frame_ratio", 0.15))
    bank = MachineReferenceBank(
        method=str(scfg["method"]),
        knn_k=int(scfg["knn_k"]),
        target_knn_k=int(scfg["target_knn_k"]),
        emb_weight=float(scfg.get("emb_score_weight", 0.55)),
        recon_weight=float(scfg.get("recon_score_weight", 0.45)),
        threshold_percentile=float(scfg["threshold_percentile"]),
    )

    report_dir = Path(cfg["paths"]["report_dir"])
    rows: List[Dict[str, Any]] = []

    for machine in sorted(train_by_machine.keys()):
        tr_list = train_by_machine[machine]
        te_list = test_by_machine[machine]

        src_clips = [c for c in tr_list if c.domain == "source"]
        tgt_clips = [c for c in tr_list if c.domain == "target"]

        src_embs, src_recs = extract_clip_features(
            src_clips, backbone, extractor, device, top_frame_ratio=top_ratio, desc=f"{machine} [train-src]"
        )
        tgt_embs, tgt_recs = None, None
        if tgt_clips:
            tgt_embs, tgt_recs = extract_clip_features(
                tgt_clips, backbone, extractor, device, top_frame_ratio=top_ratio, desc=f"{machine} [train-tgt]"
            )

        threshold = bank.fit_machine(machine, src_embs, src_recs, tgt_embs, tgt_recs)

        te_embs, te_recs = extract_clip_features(
            te_list, backbone, extractor, device, top_frame_ratio=top_ratio, desc=f"{machine} [test]"
        )
        hybrid_scores, emb_scores, rec_scores = bank.score_components(machine, te_embs, te_recs)

        y_true = np.array([c.label for c in te_list], dtype=int)
        domains = [c.domain for c in te_list]
        pauc_fpr = float(cfg["evaluation"]["pauc_max_fpr"])

        m_hyb = compute_dcase_machine_metrics(y_true, hybrid_scores, domains, pauc_max_fpr=pauc_fpr)
        m_emb = compute_dcase_machine_metrics(y_true, emb_scores, domains, pauc_max_fpr=pauc_fpr)
        m_rec = compute_dcase_machine_metrics(y_true, rec_scores, domains, pauc_max_fpr=pauc_fpr)

        rows.append(
            {
                "dsp_mode": extractor.mode,
                "machine": machine,
                "auc_source": round(m_hyb["auc_source"], 4),
                "auc_target": round(m_hyb["auc_target"], 4),
                "pauc": round(m_hyb["pauc"], 4),
                "hmean": round(m_hyb["hmean"], 4),
                "emb_only_hmean": round(m_emb["hmean"], 4),
                "recon_only_hmean": round(m_rec["hmean"], 4),
                "threshold": round(threshold, 4),
            }
        )

        if save_outputs:
            plot_machine_roc_and_scores(
                machine, y_true, hybrid_scores, threshold, report_dir / f"eval_{machine}.png"
            )

    rows.append(
        {
            "dsp_mode": extractor.mode,
            "machine": "ALL (Harmonic Mean)",
            "auc_source": round(harmonic_mean([r["auc_source"] for r in rows]), 4),
            "auc_target": round(harmonic_mean([r["auc_target"] for r in rows]), 4),
            "pauc": round(harmonic_mean([r["pauc"] for r in rows]), 4),
            "hmean": round(harmonic_mean([r["hmean"] for r in rows]), 4),
            "emb_only_hmean": round(harmonic_mean([r["emb_only_hmean"] for r in rows]), 4),
            "recon_only_hmean": round(harmonic_mean([r["recon_only_hmean"] for r in rows]), 4),
            "threshold": float("nan"),
        }
    )

    df = pd.DataFrame(rows)
    if save_outputs:
        bank.save(cfg["paths"]["reference_bank_path"])
        report_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_dir / "dcase_eval_results.csv", index=False)
        with (report_dir / "dcase_eval_results.json").open("w", encoding="utf-8") as fp:
            json.dump(rows, fp, indent=2)

    return df, bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCASE 2026 Hybrid Unsupervised ASD.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dsp-mode", type=str, default=None)
    parser.add_argument("--scoring-method", type=str, default=None, choices=["mahalanobis", "knn"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg["paths"]["dataset_root"] = args.dataset_root
    if args.scoring_method:
        cfg["scoring"]["method"] = args.scoring_method

    seed_everything(int(cfg["project"]["seed"]))
    df, _ = evaluate_dsp_pipeline(cfg, dsp_mode=args.dsp_mode, save_outputs=True)

    print("\n=== Official DCASE 2026 Unsupervised Hybrid ASD Evaluation ===")
    print(df.to_string(index=False))
    print(f"\nSaved Normal Reference Bank -> {Path(cfg['paths']['reference_bank_path']).resolve()}")


if __name__ == "__main__":
    main()
