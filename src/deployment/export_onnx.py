"""Export trained AcousticHybridBackbone (Encoder + Decoder) to ONNX & verify numerical parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.dsp.features import StereoAcousticExtractor
from src.models.ds_cnn import AcousticHybridBackbone


def infer_input_shape(cfg: Dict[str, Any]) -> Tuple[int, int, int, int]:
    extractor = StereoAcousticExtractor(cfg)
    dummy_stereo = np.zeros((2, extractor.window_samples), dtype=np.float32)
    feat = extractor.extract_single_window(dummy_stereo)
    return (1, feat.shape[0], feat.shape[1], feat.shape[2])


def load_backbone_from_checkpoint(checkpoint_path: str | Path, cfg: Dict[str, Any]) -> AcousticHybridBackbone:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mcfg = cfg["model"]
    backbone = AcousticHybridBackbone(
        in_channels=int(mcfg["in_channels"]),
        embedding_dim=int(mcfg["embedding_dim"]),
        width_mult=float(mcfg["width_mult"]),
        dropout=float(mcfg["dropout"]),
    )
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()
    return backbone


def export_backbone_to_onnx(
    checkpoint_path: str | Path,
    onnx_output_path: str | Path,
    cfg: Dict[str, Any],
    opset_version: int = 17,
) -> Path:
    """Export hybrid backbone returning (embedding, frame_mse) with dynamic batch axis."""
    backbone = load_backbone_from_checkpoint(checkpoint_path, cfg)
    input_shape = infer_input_shape(cfg)
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    out_path = Path(onnx_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        backbone,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["embedding", "frame_mse"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "embedding": {0: "batch_size"},
            "frame_mse": {0: "batch_size"},
        },
    )

    onnx.checker.check_model(onnx.load(str(out_path)))

    with torch.no_grad():
        pt_emb, pt_mse = backbone(dummy)
        pt_emb_np, pt_mse_np = pt_emb.numpy(), pt_mse.numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_emb, ort_mse = session.run(["embedding", "frame_mse"], {"spectrogram": dummy.numpy()})

    diff_emb = float(np.max(np.abs(pt_emb_np - ort_emb)))
    diff_mse = float(np.max(np.abs(pt_mse_np - ort_mse)))

    print(f"ONNX Hybrid Backbone exported to: {out_path.resolve()}")
    print(f"Input: {input_shape} -> Embedding: {ort_emb.shape}, FrameMSE: {ort_mse.shape}")
    print(f"Max abs parity diff | Embedding: {diff_emb:.6e} | FrameMSE: {diff_mse:.6e}")
    if max(diff_emb, diff_mse) > 1e-4:
        raise RuntimeError(f"ONNX numerical parity check failed: emb={diff_emb}, mse={diff_mse}")
    return out_path
