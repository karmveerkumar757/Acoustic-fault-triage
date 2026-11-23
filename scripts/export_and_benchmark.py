"""Phase 6: Export Hybrid AcousticDSCNN Backbone to ONNX & benchmark CPU latency/throughput."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deployment.benchmark import run_throughput_benchmark
from src.deployment.export_onnx import export_backbone_to_onnx
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Hybrid ONNX backbone & benchmark CPU speed.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(cfg["paths"]["checkpoint_path"])
    onnx_path = Path(cfg["paths"]["onnx_path"])

    export_backbone_to_onnx(ckpt_path, onnx_path, cfg)
    df = run_throughput_benchmark(ckpt_path, onnx_path, cfg)

    print("\n=== CPU Inference Latency & Throughput Benchmark ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
