# ─────────────────────────────────────────────────────────────────────────────
# NeuroScan AI — Backend Dockerfile
# Multi-stage build: deps are installed in builder, copied to slim runtime.
# The model weights are NOT baked into the image; mount weights/ as a volume.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

# ── Stage 2: lean runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy only the API source file
COPY main.py .

# Weights are mounted at runtime — see docker-compose.yml
VOLUME ["/weights"]

EXPOSE 8000

# Healthcheck so compose knows when the backend is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
