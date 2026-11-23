"""CPU Inference Latency (p50/p95/p99 ms) & Throughput (windows/sec) Benchmarker."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.deployment.export_onnx import infer_input_shape, load_backbone_from_checkpoint
from src.models.ds_cnn import count_parameters


def _measure(fn, warmup: int, runs: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times, dtype=np.float64)


def run_throughput_benchmark(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    bcfg = cfg["benchmark"]
    warmup = int(bcfg.get("warmup_runs", 20))
    runs = int(bcfg.get("timed_runs", 200))
    batch_sizes = [int(b) for b in bcfg.get("batch_sizes", [1, 8, 32])]

    backbone = load_backbone_from_checkpoint(checkpoint_path, cfg)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    _, c, f, t = infer_input_shape(cfg)
    rows: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        x_np = np.random.randn(bs, c, f, t).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_ms = _measure(lambda: backbone(x_pt), warmup=warmup, runs=runs)
        ort_ms = _measure(
            lambda: session.run(["embedding", "frame_mse"], {"spectrogram": x_np}),
            warmup=warmup,
            runs=runs,
        )

        for name, ms in [("PyTorch CPU", pt_ms), ("ONNX Runtime CPU", ort_ms)]:
            mean_ms = float(np.mean(ms))
            rows.append(
                {
                    "runtime": name,
                    "batch_size": bs,
                    "p50_ms": round(float(np.percentile(ms, 50)), 3),
                    "p95_ms": round(float(np.percentile(ms, 95)), 3),
                    "p99_ms": round(float(np.percentile(ms, 99)), 3),
                    "mean_ms": round(mean_ms, 3),
                    "throughput_windows_per_sec": round((bs / mean_ms) * 1000.0, 1),
                    "params": count_parameters(backbone),
                    "onnx_size_mb": round(Path(onnx_path).stat().st_size / (1024 * 1024), 3),
                }
            )

    df = pd.DataFrame(rows)
    report_dir = Path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "throughput_benchmark.csv", index=False)
    with (report_dir / "throughput_benchmark.json").open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)
    return df
