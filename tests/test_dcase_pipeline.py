"""Unit tests for Iteration 2: Spatial Dual-Mic DSP, Hybrid AE+SubCluster Model, and Hybrid Scorer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dsp.augmentations import generate_pink_factory_noise
from src.dsp.filters import dual_channel_spectral_subtraction
from src.models.ds_cnn import DCaseHybridModel
from src.scoring.anomaly_scorer import MachineReferenceBank, summarize_frame_recon_error
from src.utils.metrics import compute_dcase_machine_metrics, harmonic_mean


def test_dual_channel_spectral_subtraction_improves_snr():
    sr = 16000
    n = sr * 2
    t = np.arange(n) / float(sr)
    rng = np.random.default_rng(7)
    target = np.sin(2.0 * np.pi * 600.0 * t).astype(np.float32)
    ambient = generate_pink_factory_noise(n, sr, rng)

    near = target + 1.0 * ambient
    far = 0.1 * target + 1.0 * ambient
    cleaned = dual_channel_spectral_subtraction(near, far, sr=sr, alpha=0.8, beta=0.05)

    assert cleaned.shape == near.shape
    assert float(np.corrcoef(target, cleaned)[0, 1]) > float(np.corrcoef(target, near)[0, 1])


def test_hybrid_model_shapes_and_edge_forward():
    model = DCaseHybridModel(in_channels=3, embedding_dim=256, num_classes=6, sub_clusters=8)
    x = torch.randn(4, 3, 128, 94)

    emb, recon, logits = model(x)
    assert emb.shape == (4, 256)
    assert recon.shape == (4, 3, 128, 94)
    assert logits.shape == (4, 6)
    assert torch.allclose(torch.norm(emb, p=2, dim=1), torch.ones(4), atol=1e-4)

    edge_emb, frame_mse = model.backbone(x)
    assert edge_emb.shape == (4, 256)
    assert frame_mse.shape == (4, 94)


def test_hybrid_scorer_and_dcase_metrics(tmp_path):
    rng = np.random.default_rng(42)
    src = rng.normal(0, 0.08, size=(60, 32)).astype(np.float32); src[:, 0] += 1.0
    tgt = rng.normal(0, 0.08, size=(10, 32)).astype(np.float32); tgt[:, 1] += 1.0
    src_rec = rng.uniform(0.01, 0.03, size=60)
    tgt_rec = rng.uniform(0.01, 0.03, size=10)

    bank = MachineReferenceBank(method="mahalanobis", emb_weight=0.55, recon_weight=0.45)
    bank.fit_machine("bearingEmu", src, src_rec, tgt, tgt_rec)
    reloaded = MachineReferenceBank.load(bank.save(tmp_path / "bank.npz"))

    te_s = rng.normal(0, 0.08, size=(10, 32)).astype(np.float32); te_s[:, 0] += 1.0
    te_t = rng.normal(0, 0.08, size=(10, 32)).astype(np.float32); te_t[:, 1] += 1.0
    te_a = rng.normal(0, 0.08, size=(20, 32)).astype(np.float32); te_a[:, 2] += 1.0
    te_recs = np.concatenate([rng.uniform(0.01, 0.03, 20), rng.uniform(0.08, 0.15, 20)])

    hybrid, _, _ = reloaded.score_components("bearingEmu", np.vstack([te_s, te_t, te_a]), te_recs)
    y_true = np.array([0] * 20 + [1] * 20)
    domains = ["source"] * 10 + ["target"] * 10 + ["source"] * 10 + ["target"] * 10
    m = compute_dcase_machine_metrics(y_true, hybrid, domains)
    assert m["auc_source"] > 0.99 and m["auc_target"] > 0.99 and m["hmean"] > 0.98
    assert summarize_frame_recon_error(np.array([0.01, 0.01, 0.50]), top_ratio=0.33) > 0.25
    assert harmonic_mean([0.8, 0.8, 0.8]) == 0.8
