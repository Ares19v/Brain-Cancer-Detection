# EVAL — NeuroScan AI (Brain Cancer Detection)

> **Evaluation Date:** 2026-05-29
> **Evaluator:** Automated Portfolio Review
> **Maturity Level:** Production-Ready (Maturity Score: 8.5/10)

---

## 1. Project Purpose & Problem Statement

NeuroScan AI is a full-stack, deep-learning-powered medical diagnostic interface that automates the classification of brain MRI scans. It solves the critical bottleneck of primary radiologist triage by identifying and sorting scans into four distinct categories in under a second: **Glioma**, **Meningioma**, **Pituitary**, or **Healthy**.

To enforce responsible clinical outcomes, the platform employs a safety-critical confidence filter: any classification yielding less than **75% confidence** is automatically categorized as **INCONCLUSIVE** rather than outputting a potential misdiagnosis.

---

## 2. Technical Architecture

NeuroScan AI follows a decoupled microservices design:

- **Machine Learning Core (PyTorch + `timm`):**
  - Fine-tuned **EfficientNet-B0** trained on the Brain Tumour MRI Dataset (7,023 high-resolution scans).
  - Validation accuracy reaches **95.6%** (Internal Top-1) and **83.3%** on external unseen datasets.
  - Normalization and pre-processing pipeline handles resizing (`224x224`), tensor conversion, and ImageNet standardization.
- **Backend API Layer (FastAPI + Python 3.11):**
  - Serves predictions via high-concurrency `/predict` and `/health` HTTP endpoints.
  - Incorporates graceful 503 error degradation if weights are missing, allowing CI checks to compile syntax-only pipelines safely.
- **Frontend Dashboard (React 19 + Vite):**
  - Seamless drag-and-drop file upload with real-time preview.
  - Interactive result cards highlighting classification states, warning cards, and patient metadata.
  - PDF diagnostic dossier generation via `jsPDF`.
- **Infrastructure:** Fully containerized with multi-stage Dockerfiles orchestrated via Docker Compose.

---

## 3. Hugging Face Weights Integration

The project has its production-ready model published publicly on Hugging Face:

- **Repository:** `devanshty/brain-cancer-detection`
- **Download Instructions:**
  ```python
  from huggingface_hub import hf_hub_download
  model_path = hf_hub_download(repo_id='devanshty/brain-cancer-detection', filename='brain_model.pth')
  ```
  Place the downloaded weights at `weights/efficientnet_b0.pth` before booting the FastAPI microservice.

---

## 4. Strengths

- **High-Performance Architecture (EfficientNet-B0):** Choosing a lightweight yet feature-rich CNN like EfficientNet-B0 guarantees sub-second local CPU/GPU execution without losing accuracy.
- **Clinical Safety Thresholding:** Gating outputs at a `75%` confidence barrier is a great display of security and clinical ethics thinking.
- **Polished UX & PDF Dossier:** Clean design with custom patient name tags, drag-and-drop file wrappers, and rapid exportable reports.
- **Graceful CI Initialization:** Separation of model instantiation inside `CI_MODE` checks prevents GitHub Actions runners from crashing due to binary weight absence.

---

## 5. Limitations & Known Gaps

- **Binary Classification Confusion:** Medical imagery (like a knee MRI or a chest X-Ray) uploaded accidentally will still be processed through the model. While the 75% threshold mitigates this, a pre-classification binary filter (is this a brain MRI or not?) would improve structural security.
- **GPU Inference Latency Overhead:** For multiple parallel uploads, PyTorch execution runs synchronously on the main event thread, which could block FastAPI async event loops.
- **No Active Segmentation:** The system classifies the *presence* of a tumor but does not isolate, mask, or outline the tumor region (no U-Net or YOLOv8-seg integration).

---

## 6. Code Quality Assessment

- **Clean Decoupling:** Complete segregation of FastAPI routes, training scripts, Docker configs, and React modules.
- **Extensive Pytest Config:** Training, validation, and confusion matrix builders in `scripts/` are highly structured and documented.

---

## 7. Maturity Breakdown

| Dimension | Score | Notes |
|-----------|-------|-------|
| Functionality | 8/10 | High-accuracy classification with robust safety gating. |
| Code Quality | 9/10 | Solid PyTorch practices, clean transform pipelines, and beautiful frontend UI code. |
| Documentation | 8/10 | Explicit Hugging Face download guides and setup parameters. |
| Scalability | 7/10 | Synchronous PyTorch execution can bottleneck under high concurrency. |
| Security | 9/10 | Secure non-root container structures and robust validation checks. |
| **Overall** | **8.5/10** | **Highly functional medical classification demo.** Showcases professional AI integration. |

---

## 8. Suggested Next Steps

1. **Add Binary Classifier Filter:** Implement a small pre-classifier network to verify if the uploaded file is indeed a transverse/sagittal brain MRI scan before feeding it to EfficientNet.
2. **Implement PyTorch Batch Threading:** Run PyTorch forward passes inside an asynchronous thread executor (`anyio.to_thread`) to prevent blocking Uvicorn's event loop during concurrent requests.
3. **Tumor Segmentation:** Integrate a lightweight U-Net or YOLOv11-seg model to draw a bounding contour around detected tumors in the MRI image, providing direct visual confirmation.

---

<p align="center">Made by Devansh Tyagi @ 2026</p>
