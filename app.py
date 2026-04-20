"""
OrthoAI Demo App — FastAPI + Custom HTML Frontend
=================================================
Phase B  — FDI instance labeling   (YOLOv8x-seg,      mAP50=0.956)
Phase D  — Disease classification  (EfficientNetV2-M, F1=0.8927)
XAI      — GradCAM per diseased tooth

Run:
    python app.py
    python app.py --device cuda | mps | cpu
    python app.py --port 7860
"""
from __future__ import annotations

import argparse
import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.device import get_device
from inference.models import ModelRegistry
from inference.panoramic import run_panoramic
from inference.cephalometric import run_cephalometric_structured

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE   = Path(__file__).parent
MDL_DIR = _HERE / "DATA" / "processed" / "models_v4"
SSL_PATH = _HERE / "DATA" / "processed" / "backbone_ssl_v2.pth"
STATIC  = _HERE / "static"
STATIC.mkdir(exist_ok=True)

# ─── Global model registry ────────────────────────────────────────────────────

REGISTRY: ModelRegistry | None = None
DEVICE_NAME: str = "auto"


def _load_models() -> None:
    global REGISTRY
    if REGISTRY is not None:
        return
    device = get_device(DEVICE_NAME)
    REGISTRY = ModelRegistry(MDL_DIR, ssl_path=SSL_PATH, device=device)
    REGISTRY.load_all(skip_phase_a=True)


# ─── Image encoding ───────────────────────────────────────────────────────────

def _to_jpeg_b64(img_rgb: np.ndarray, quality: int = 88) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="OrthoAI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>static/index.html not found</h2>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/capabilities")
async def capabilities() -> JSONResponse:
    _load_models()
    return JSONResponse({
        "panoramic":    REGISTRY is not None and REGISTRY.yolo_model is not None,
        "cephalometric": REGISTRY is not None and REGISTRY.ceph_model is not None,
        "cvm":          REGISTRY is not None and REGISTRY.cvm_model is not None,
    })


@app.post("/analyze_ceph")
async def analyze_ceph(image: UploadFile = File(...)) -> JSONResponse:
    _load_models()
    if REGISTRY is None or REGISTRY.ceph_model is None:
        return JSONResponse(
            {"error": "Cephalometric model not available."}, status_code=503)

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = run_cephalometric_structured(tmp_path, REGISTRY)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return JSONResponse({
        "landmark_image":     _to_jpeg_b64(result["landmark_image"], quality=90),
        "landmarks":          result["landmarks"],
        "cvm_stage":          result["cvm_stage"],
        "cvm_label":          result["cvm_label"],
        "cvm_probs":          result["cvm_probs"],
        "cvm_interpretation": result["cvm_interpretation"],
    })


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    _load_models()
    if REGISTRY is None or REGISTRY.yolo_model is None:
        return JSONResponse({"error": "Models not loaded"}, status_code=503)

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        annotated, results, crops = run_panoramic(tmp_path, REGISTRY, progress_cb=None)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Attach GradCAM b64 to each diseased result (order matches crops list)
    crop_idx = 0
    for r in results:
        if r["disease"] != "Healthy" and crop_idx < len(crops):
            r["gradcam_b64"] = _to_jpeg_b64(crops[crop_idx], quality=82)
            crop_idx += 1
        else:
            r["gradcam_b64"] = None

    return JSONResponse({
        "annotated": _to_jpeg_b64(annotated, quality=90),
        "results": [
            {
                "fdi":        r["fdi"],
                "disease":    r["disease"],
                "confidence": round(float(r["confidence"]), 4),
                "gradcam_b64": r.get("gradcam_b64"),
            }
            for r in results
        ],
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    global DEVICE_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    DEVICE_NAME = args.device
    print(f"[OrthoAI] Loading models on device={DEVICE_NAME} ...")
    _load_models()
    print(f"[OrthoAI] Ready → http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
