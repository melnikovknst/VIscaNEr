# syntax=docker/dockerfile:1

# 1. The web interface: a static React build.
FROM node:24-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json tsconfig.json vite.config.ts ./
RUN npm ci --no-audit --no-fund
COPY frontend ./frontend
RUN npm run build

# 2. The API with the recognition pipeline; it also serves the web interface.
FROM python:3.13-slim
# cu128 wheels run on any recent NVIDIA driver and fall back to CPU without a GPU;
# run.sh passes the much smaller CPU-only index when no GPU is available.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY five_stream_transformer ./five_stream_transformer
COPY joint_yolo ./joint_yolo
COPY data ./data
COPY models ./models
COPY --from=frontend /app/dist ./dist

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import json,urllib.request as u; exit(0 if json.load(u.urlopen('http://127.0.0.1:8080/api/health'))['model_ready'] else 1)"
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
