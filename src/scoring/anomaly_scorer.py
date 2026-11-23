"""Unsupervised Domain-Generalized Anomaly Scorer (Prototype-Balanced 1-NN, kNN, Mahalanobis)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    return (x / norms).astype(np.float32)


def _cosine_knn_distances(query: np.ndarray, ref: np.ndarray, k: int, leave_one_out: bool = False) -> np.ndarray:
    q = _l2_normalize(query)
    r = _l2_normalize(ref)
    dist = 1.0 - np.clip(np.matmul(q, r.T), -1.0, 1.0)
    if leave_one_out and dist.shape[0] == dist.shape[1]:
        np.fill_diagonal(dist, np.inf)
    k_eff = max(1, min(k, r.shape[0] - (1 if leave_one_out else 0)))
    if k_eff == 1:
        return np.min(dist, axis=1)
    part = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.mean(part, axis=1)


def summarize_frame_recon_error(frame_mse: np.ndarray, top_ratio: float = 0.15) -> float:
    """Aggregate per-frame reconstruction MSE into a single clip score (mean + top-k% peak)."""
    flat = np.asarray(frame_mse, dtype=np.float64).reshape(-1)
    if len(flat) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(flat) * top_ratio)))
    top_mean = float(np.mean(np.sort(flat)[-k:]))
    return 0.5 * float(np.mean(flat)) + 0.5 * top_mean


class MachineReferenceBank:
    """Multi-machine normal reference bank with Prototype-Balanced 1-NN (DCASE winner)."""

    def __init__(
        self,
        method: str = "proto_1nn",
        knn_k: int = 5,
        target_knn_k: int = 1,
        proto_clusters: int = 16,
        emb_weight: float = 1.0,
        recon_weight: float = 0.0,
        threshold_percentile: float = 92.0,
    ):
        self.method = method.lower()
        self.knn_k = int(knn_k)
        self.target_knn_k = int(target_knn_k)
        self.proto_clusters = int(proto_clusters)
        self.emb_weight = float(emb_weight)
        self.recon_weight = float(recon_weight)
        self.threshold_percentile = float(threshold_percentile)

        self.machines: Dict[str, Dict[str, np.ndarray]] = {}
        self.stats: Dict[str, Dict[str, float]] = {}
        self.thresholds: Dict[str, float] = {}
        self._mahal_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _fit_mahalanobis(self, machine: str, src: np.ndarray) -> None:
        mean = np.mean(src, axis=0).astype(np.float32)
        lw = LedoitWolf().fit(src)
        self._mahal_cache[machine] = {"mean": mean, "precision": lw.precision_.astype(np.float32)}

    def _mahalanobis_dist(self, machine: str, query: np.ndarray) -> np.ndarray:
        if machine not in self._mahal_cache:
            self._fit_mahalanobis(machine, self.machines[machine]["source"])
        mean = self._mahal_cache[machine]["mean"]
        prec = self._mahal_cache[machine]["precision"]
        diff = _l2_normalize(query) - mean
        md2 = np.einsum("ij,jk,ik->i", diff, prec, diff)
        return np.sqrt(np.maximum(md2, 0.0))

    def fit_machine(
        self,
        machine: str,
        source_embeddings: np.ndarray,
        source_recon_errors: np.ndarray,
        target_embeddings: np.ndarray | None = None,
        target_recon_errors: np.ndarray | None = None,
    ) -> float:
        """Fit normal reference bank & calibrate alert threshold strictly from train/ normals."""
        src = _l2_normalize(np.asarray(source_embeddings, dtype=np.float32))
        entry: Dict[str, np.ndarray] = {"source": src}
        if target_embeddings is not None and len(target_embeddings) > 0:
            entry["target"] = _l2_normalize(np.asarray(target_embeddings, dtype=np.float32))

        if self.method == "proto_1nn":
            k_proto = min(self.proto_clusters, len(src))
            km = KMeans(n_clusters=k_proto, n_init=10, random_state=42).fit(src)
            entry["prototypes"] = _l2_normalize(km.cluster_centers_.astype(np.float32))
            self.machines[machine] = entry
            all_proto = (
                np.vstack([entry["prototypes"], entry["target"]])
                if "target" in entry
                else entry["prototypes"]
            )
            train_emb_scores = _cosine_knn_distances(src, all_proto, k=1, leave_one_out=False)
        elif self.method == "mahalanobis":
            self.machines[machine] = entry
            self._fit_mahalanobis(machine, src)
            train_emb_scores = self._mahalanobis_dist(machine, src)
        else:
            self.machines[machine] = entry
            train_emb_scores = _cosine_knn_distances(src, src, k=self.knn_k, leave_one_out=True)

        src_rec = np.asarray(source_recon_errors, dtype=np.float64)
        all_rec = (
            np.concatenate([src_rec, np.asarray(target_recon_errors, dtype=np.float64)])
            if target_recon_errors is not None and len(target_recon_errors) > 0
            else src_rec
        )

        emb_med = float(np.median(train_emb_scores))
        emb_iqr = float(np.percentile(train_emb_scores, 75) - np.percentile(train_emb_scores, 25)) + 1e-6
        rec_med = float(np.median(all_rec))
        rec_iqr = float(np.percentile(all_rec, 75) - np.percentile(all_rec, 25)) + 1e-6

        self.stats[machine] = {
            "emb_med": emb_med,
            "emb_iqr": emb_iqr,
            "rec_med": rec_med,
            "rec_iqr": rec_iqr,
        }

        if self.recon_weight <= 0.0:
            train_final = train_emb_scores
        else:
            z_emb = (train_emb_scores - emb_med) / emb_iqr
            z_rec = (src_rec - rec_med) / rec_iqr
            train_final = self.emb_weight * z_emb + self.recon_weight * z_rec

        thresh = float(np.percentile(train_final, self.threshold_percentile))
        self.thresholds[machine] = thresh
        return thresh

    def score_components(
        self,
        machine: str,
        embeddings: np.ndarray,
        recon_errors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (final_scores, emb_raw_scores, recon_raw_scores)."""
        if machine not in self.machines:
            raise KeyError(f"Machine '{machine}' not in bank. Available: {self.list_machines()}")

        q = np.asarray(embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        rec = np.asarray(recon_errors, dtype=np.float64).reshape(-1)
        entry = self.machines[machine]

        if self.method == "proto_1nn":
            all_proto = (
                np.vstack([entry["prototypes"], entry["target"]])
                if "target" in entry
                else entry["prototypes"]
            )
            emb_scores = _cosine_knn_distances(q, all_proto, k=1, leave_one_out=False)
        elif self.method == "knn":
            d_src = _cosine_knn_distances(q, entry["source"], k=self.knn_k, leave_one_out=False)
            if "target" in entry and len(entry["target"]) > 0:
                d_tgt = _cosine_knn_distances(q, entry["target"], k=self.target_knn_k, leave_one_out=False)
                emb_scores = np.minimum(d_src, d_tgt)
            else:
                emb_scores = d_src
        else:
            d_src = self._mahalanobis_dist(machine, q)
            if "target" in entry and len(entry["target"]) > 0:
                d_tgt_cos = _cosine_knn_distances(q, entry["target"], k=self.target_knn_k, leave_one_out=False)
                src_cos_med = float(
                    np.median(_cosine_knn_distances(entry["source"], entry["source"], k=self.knn_k, leave_one_out=True))
                    + 1e-6
                )
                emb_scores = np.minimum(d_src, d_tgt_cos * (self.stats[machine]["emb_med"] / src_cos_med))
            else:
                emb_scores = d_src

        if self.recon_weight <= 0.0:
            return emb_scores, emb_scores, rec

        st = self.stats[machine]
        z_emb = (emb_scores - st["emb_med"]) / st["emb_iqr"]
        z_rec = (rec - st["rec_med"]) / st["rec_iqr"]
        hybrid = self.emb_weight * z_emb + self.recon_weight * z_rec
        return hybrid, emb_scores, rec

    def get_threshold(self, machine: str) -> float:
        return float(self.thresholds[machine])

    def list_machines(self) -> List[str]:
        return sorted(self.machines.keys())

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, np.ndarray] = {}
        meta = {
            "method": self.method,
            "knn_k": self.knn_k,
            "target_knn_k": self.target_knn_k,
            "proto_clusters": self.proto_clusters,
            "emb_weight": self.emb_weight,
            "recon_weight": self.recon_weight,
            "threshold_percentile": self.threshold_percentile,
            "stats": self.stats,
            "thresholds": self.thresholds,
            "machines": sorted(self.machines.keys()),
        }
        payload["__meta__"] = np.array(json.dumps(meta))
        for m, entry in self.machines.items():
            payload[f"{m}__source"] = entry["source"]
            if "target" in entry:
                payload[f"{m}__target"] = entry["target"]
            if "prototypes" in entry:
                payload[f"{m}__prototypes"] = entry["prototypes"]
        np.savez_compressed(out, **payload)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "MachineReferenceBank":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Reference bank not found: {p.resolve()}. Run `python scripts/evaluate_dcase.py` first."
            )
        data = np.load(p, allow_pickle=False)
        meta = json.loads(str(data["__meta__"].item()))
        bank = cls(
            method=meta["method"],
            knn_k=meta["knn_k"],
            target_knn_k=meta["target_knn_k"],
            proto_clusters=meta.get("proto_clusters", 16),
            emb_weight=meta["emb_weight"],
            recon_weight=meta["recon_weight"],
            threshold_percentile=meta["threshold_percentile"],
        )
        bank.stats = meta["stats"]
        bank.thresholds = {k: float(v) for k, v in meta["thresholds"].items()}
        for m in meta["machines"]:
            entry = {"source": data[f"{m}__source"].astype(np.float32)}
            if f"{m}__target" in data:
                entry["target"] = data[f"{m}__target"].astype(np.float32)
            if f"{m}__prototypes" in data:
                entry["prototypes"] = data[f"{m}__prototypes"].astype(np.float32)
            bank.machines[m] = entry
        return bank