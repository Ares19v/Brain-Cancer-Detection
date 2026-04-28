"""
NeuroScan AI — Brain Tumour Classification API
================================================
FastAPI backend that serves the EfficientNet-B0 classification model.

Endpoints:
  GET  /health   — liveness check (used by Docker healthcheck & CI)
  POST /predict  — accepts an MRI image, returns diagnosis + confidence

Classes: GLIOMA | MENINGIOMA | HEALTHY | PITUITARY
Safety threshold: predictions below 75% confidence are returned as INCONCLUSIVE.
"""

from __future__ import annotations

import io
import os
import sys

import torch
import timm
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_BASE_DIR, "..", "weights", "efficientnet_b0.pth")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES     = ["GLIOMA", "MENINGIOMA", "HEALTHY", "PITUITARY"]
CONF_THRESH = 0.75
CI_MODE     = os.getenv("CI", "").lower() in ("true", "1", "yes")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroScan AI",
    description="AI-powered brain tumour classification from MRI scans.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model loading (skipped in CI to allow syntax-only checks) ─────────────────
_model: torch.nn.Module | None = None

def _load_model() -> torch.nn.Module:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at: {MODEL_PATH}\n"
            "Download efficientnet_b0.pth and place it in the weights/ directory."
        )
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


if not CI_MODE:
    try:
        _model = _load_model()
        print(f"[NeuroScan] Model loaded on {DEVICE}  |  Classes: {CLASSES}")
    except FileNotFoundError as e:
        print(f"[NeuroScan] WARNING: {e}", file=sys.stderr)
        print("[NeuroScan] Server will start, but /predict will return 503 until weights are present.", file=sys.stderr)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health() -> dict:
    """Liveness + readiness probe."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Classify a brain MRI image.

    - **file**: JPEG/PNG brain MRI scan
    - Returns diagnosis, confidence (%), and status.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights not loaded. Place efficientnet_b0.pth in weights/ and restart.",
        )

    # ── Read & decode image ───────────────────────────────────────────────────
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid JPEG or PNG.")

    # ── Inference ─────────────────────────────────────────────────────────────
    input_tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs       = _model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    conf_score = confidence.item()
    diagnosis  = CLASSES[predicted_idx.item()]

    # ── Safety filter ─────────────────────────────────────────────────────────
    if conf_score < CONF_THRESH:
        return {
            "diagnosis":  "INCONCLUSIVE",
            "confidence": round(conf_score * 100, 2),
            "status":     "Inconclusive — Not a clear Brain MRI",
            "warning":    "Low confidence score. Please upload a clear, high-quality Brain MRI scan.",
        }

    status = "Healthy" if diagnosis == "HEALTHY" else "Tumor Detected"

    return {
        "diagnosis":  diagnosis,
        "confidence": round(conf_score * 100, 2),
        "status":     status,
    }


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
