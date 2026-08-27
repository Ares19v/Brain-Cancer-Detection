# NeuroScan AI 🧠

<div align="center">

**AI-powered brain MRI classification using EfficientNet-B0**

[![CI](https://github.com/Ares19v/brain-cancer-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Ares19v/brain-cancer-detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)


![NeuroScan Demo](outputs/confusion_matrix.png)

</div>

---

## What It Does

NeuroScan analyses brain MRI scans and classifies them into one of four categories in under a second:

| Label | Description |
|---|---|
| **GLIOMA** | Glial cell tumour — most common primary brain tumour |
| **MENINGIOMA** | Tumour arising from the meninges — often benign |
| **PITUITARY** | Tumour of the pituitary gland |
| **HEALTHY** | No tumour detected |

The model returns a **confidence score** with every prediction. Results below 75% confidence are flagged as **INCONCLUSIVE** rather than guessing, ensuring responsible clinical output.

---

## Architecture

```
brain-cancer-detection/
├── backend/            FastAPI inference server (Python 3.11)
│   ├── main.py         /health + /predict endpoints
│   └── Dockerfile      Multi-stage Docker image
├── frontend/           React 19 + Vite SPA
│   ├── src/App.jsx     Drag-drop upload, results, PDF report
│   └── Dockerfile      Nginx multi-stage image
├── weights/            Model weights (download — see below)
├── scripts/            Training, validation, and analysis utilities
├── datasets/           Brain MRI dataset (not tracked in git)
├── docker-compose.yml  Orchestrate both services
├── Run_Project.bat     One-click Windows launcher
└── INSTALL.bat         First-time dependency installer
```

**Model:** EfficientNet-B0 fine-tuned on the [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,023 MRI images).

**Validation Stats:**
- **Internal Accuracy:** 95.6% (Top-1)
- **External Accuracy:** 83.3% (Exact Match on 30 random unseen samples)
- **Generalization:** Proven robust on distinct datasets (SartajBhuvaji).

**Training:** AdamW optimizer, weighted cross-entropy loss, 20 epochs on NVIDIA Blackwell RTX 5060.

---

## Quick Start

### Option 1 — Windows (Recommended)

```
1. Download model weights (see below)
2. Double-click INSTALL.bat   ← sets up Python venv + npm
3. Double-click Run_Project.bat ← opens both servers + browser
```

App opens at **http://localhost:5173**

### Option 2 — Docker

```bash
# 1. Download model weights (see below)
# 2. Start both containers
docker compose up --build

# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
# API docs → http://localhost:8000/docs
```

### Option 3 — Manual

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## Model Weights

The trained weights file (`weights/efficientnet_b0.pth`, ~16 MB) is hosted on Hugging Face:

> **[⬇ Download efficientnet_b0.pth on Hugging Face](https://huggingface.co/devanshty/brain-cancer-detection/blob/main/efficientnet_b0.pth)**

Or via Python:
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="devanshty/brain-cancer-detection", filename="efficientnet_b0.pth")
```

Place the downloaded file at: `weights/efficientnet_b0.pth`


---

## API Reference

### `GET /health`
Returns server status and whether the model is loaded.

```json
{ "status": "ok", "model_loaded": true, "device": "cuda" }
```

### `POST /predict`
Upload a brain MRI image to classify it.

**Request:** `multipart/form-data` with field `file` (JPEG/PNG)

**Response:**
```json
{
  "diagnosis":  "GLIOMA",
  "confidence": 97.43,
  "status":     "Tumor Detected"
}
```

Full interactive docs at **http://localhost:8000/docs**

---

## Features

- **Drag-and-drop MRI upload** with live preview
- **Confidence-gated predictions** — returns `INCONCLUSIVE` if confidence < 75%
- **PDF report generation** — one-click download with patient name + diagnosis table
- **Zero-config launcher** — `Run_Project.bat` handles everything on Windows
- **Docker support** — spin up the full stack with one command
- **REST API** — FastAPI with auto-generated OpenAPI docs

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | EfficientNet-B0 via `timm` |
| Training | PyTorch 2.x, AdamW, weighted cross-entropy |
| Backend | FastAPI, Uvicorn |
| Frontend | React 19, Vite, Lucide, jsPDF |
| Container | Docker, nginx |
| CI/CD | GitHub Actions |

---

## Disclaimer

> NeuroScan is an academic/portfolio project. It is **not** a medical device and **must not** be used for clinical diagnosis. Always consult a qualified radiologist.

---

<p align="center">
  Made by Devansh Tyagi @ 2026
</p>

## 🤗 Model on Hugging Face

The trained model is available on Hugging Face: [devanshty/brain-cancer-detection](https://huggingface.co/devanshty/brain-cancer-detection)

### Download

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id='devanshty/brain-cancer-detection', filename='brain_model.pth')
```

---

© 2025 Devansh Tyagi (Ares19v). All Rights Reserved.

Unauthorized copying, modification, distribution, or use of this project or any of its components, in whole or in part, without explicit written permission from the author is strictly prohibited.
