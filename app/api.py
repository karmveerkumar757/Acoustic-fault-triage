"""Phase 6 FastAPI Microservice for High-Throughput Hybrid Acoustic Fault Triage."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deployment.inference import AcousticTriageEngine
from src.utils.config import load_config

app = FastAPI(
    title="AcousticPulse v2 DCASE 2026 Fault Triage API",
    version="2.0.0",
    description="2-Channel Spatial DSP + Hybrid DS-CNN ONNX Runtime + Mahalanobis/Recon Scorer.",
)

_cfg = load_config("configs/config.yaml")
_engine: Optional[AcousticTriageEngine] = None


def get_engine() -> AcousticTriageEngine:
    global _engine
    if _engine is None:
        _engine = AcousticTriageEngine(_cfg)
    return _engine


@app.get("/health")
def health_check():
    eng = get_engine()
    return {"status": "ok", "machines": eng.list_machines(), "onnx": _cfg["paths"]["onnx_path"]}


@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
    machine: str = Query(..., description="Machine type registered in the normal reference bank"),
    dsp_mode: str = Query(
        "spatial_dual_mic",
        pattern="^(spatial_dual_mic|spatial_dual_mic_gated|single_mic_gated|raw_ch0)$",
    ),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    contents = await file.read()
    try:
        res = get_engine().predict_source(io.BytesIO(contents), machine=machine, dsp_mode=dsp_mode)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "machine": res["machine"],
        "label": res["label"],
        "is_anomaly": res["is_anomaly"],
        "anomaly_score": res["anomaly_score"],
        "threshold": res["threshold"],
        "embedding_distance": res["embedding_distance"],
        "reconstruction_error": res["reconstruction_error"],
        "dsp_mode": res["dsp_mode"],
        "is_stereo": res["is_stereo"],
        "dsp_latency_ms": res["dsp_latency_ms"],
        "inference_latency_ms": res["inference_latency_ms"],
        "total_latency_ms": res["total_latency_ms"],
    }
