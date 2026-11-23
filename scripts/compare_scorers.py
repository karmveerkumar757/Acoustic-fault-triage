"""Zero-retraining benchmark of 4 unsupervised domain-generalized scorers on best_hybrid_model.pt."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_dcase import extract_clip_features
from src.data.dcase_dataset import load_dcase_manifest
from src.deployment.export_onnx import load_backbone_from_checkpoint
from src.dsp.features import StereoAcousticExtractor
from src.utils.config import load_config, resolve_device, seed_everything
from src.utils.metrics import compute_dcase_machine_metrics, harmonic_mean


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def cosine_knn(q: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    dist = 1.0 - np.clip(np.matmul(l2_norm(q), l2_norm(ref).T), -1.0, 1.0)
    k_eff = max(1, min(k, len(ref)))
    if k_eff == 1:
        return np.min(dist, axis=1)
    return np.mean(np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff], axis=1)


def score_all_methods(src: np.ndarray, tgt: np.ndarray | None, te: np.ndarray) -> dict[str, np.ndarray]:
    src = l2_norm(src)
    te = l2_norm(te)
    tgt = l2_norm(tgt) if (tgt is not None and len(tgt) > 0) else None
    out = {}

    # 1. Pure Cosine kNN (k=5 src, k=1 tgt)
    d_s = cosine_knn(te, src, k=5)
    out["1_cosine_knn"] = np.minimum(d_s, cosine_knn(te, tgt, k=1)) if tgt is not None else d_s

    # 2. Prototype-Balanced 1-NN (KMeans-16 source + all target clips)
    km16 = KMeans(n_clusters=min(16, len(src)), n_init=10, random_state=42).fit(src)
    proto16 = l2_norm(km16.cluster_centers_)
    all_proto = np.vstack([proto16, tgt]) if tgt is not None else proto16
    out["2_proto16_1nn"] = cosine_knn(te, all_proto, k=1)

    # 3. Shared-Precision Mahalanobis (single source mean + target mean)
    prec = LedoitWolf().fit(src).precision_.astype(np.float32)
    centers_s = [src.mean(axis=0)]
    if tgt is not None:
        centers_s.append(tgt.mean(axis=0))
    c_arr = np.stack(centers_s, axis=0)
    diff = te[:, None, :] - c_arr[None, :, :]
    md2 = np.einsum("nmd,dk,nmk->nm", diff, prec, diff)
    out["3_shared_mahal"] = np.sqrt(np.maximum(np.min(md2, axis=1), 0.0))

    # 4. Multi-Mode Shared-Precision Mahalanobis (K=8 source modes + all target clips)
    km8 = KMeans(n_clusters=min(8, len(src)), n_init=10, random_state=42).fit(src)
    residuals = src - km8.cluster_centers_[km8.labels_]
    prec_w = LedoitWolf().fit(residuals).precision_.astype(np.float32)
    centers_m = np.vstack([km8.cluster_centers_, tgt]) if tgt is not None else km8.cluster_centers_
    diff_m = te[:, None, :] - centers_m[None, :, :]
    md2_m = np.einsum("nmd,dk,nmk->nm", diff_m, prec_w, diff_m)
    out["4_multimode_mahal"] = np.sqrt(np.maximum(np.min(md2_m, axis=1), 0.0))

    return out


def main() -> None:
    cfg = load_config("configs/config.yaml")
    seed_everything(int(cfg["project"]["seed"]))
    device = resolve_device(str(cfg["project"]["device"]))

    backbone = load_backbone_from_checkpoint(cfg["paths"]["checkpoint_path"], cfg).to(device)
    backbone.eval()
    extractor = StereoAcousticExtractor(cfg)
    train_clips, test_clips, _ = load_dcase_manifest(cfg["paths"]["dataset_root"])

    tr_map, te_map = defaultdict(list), defaultdict(list)
    for c in train_clips:
        tr_map[c.machine].append(c)
    for c in test_clips:
        te_map[c.machine].append(c)

    rows = []
    for machine in sorted(tr_map.keys()):
        src_clips = [c for c in tr_map[machine] if c.domain == "source"]
        tgt_clips = [c for c in tr_map[machine] if c.domain == "target"]
        src_e, _ = extract_clip_features(src_clips, backbone, extractor, device, desc=f"{machine} src")
        tgt_e = extract_clip_features(tgt_clips, backbone, extractor, device, desc=f"{machine} tgt")[0] if tgt_clips else None
        te_e, _ = extract_clip_features(te_map[machine], backbone, extractor, device, desc=f"{machine} test")

        y_true = np.array([c.label for c in te_map[machine]], dtype=int)
        domains = [c.domain for c in te_map[machine]]

        for name, scores in score_all_methods(src_e, tgt_e, te_e).items():
            m = compute_dcase_machine_metrics(y_true, scores, domains, pauc_max_fpr=0.1)
            rows.append({
                "scorer": name,
                "machine": machine,
                "auc_source": round(m["auc_source"], 4),
                "auc_target": round(m["auc_target"], 4),
                "pauc": round(m["pauc"], 4),
                "hmean": round(m["hmean"], 4),
            })

    df = pd.DataFrame(rows)
    for name, grp in df.groupby("scorer"):
        print(f"\n=== Scorer: {name} ===")
        print(grp.drop(columns=["scorer"]).to_string(index=False))
        print(
            f"OVERALL HARMONIC MEAN -> "
            f"Source: {harmonic_mean(grp['auc_source'].tolist()):.4f} | "
            f"Target: {harmonic_mean(grp['auc_target'].tolist()):.4f} | "
            f"pAUC: {harmonic_mean(grp['pauc'].tolist()):.4f} | "
            f"Omega: {harmonic_mean(grp['hmean'].tolist()):.4f}"
        )


if __name__ == "__main__":
    main()