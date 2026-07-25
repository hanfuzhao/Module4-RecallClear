# Container image for the deployed RecallClear app (Hugging Face Spaces, Docker SDK).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=2 \
    PORT=7860

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY main.py ./
COPY scripts ./scripts
COPY templates ./templates
COPY static ./static
COPY models/adapter ./models/adapter
COPY data/processed/test.jsonl ./data/processed/test.jsonl
COPY data/outputs/evaluation.json ./data/outputs/evaluation.json

# Spaces runs as a non-root user; make the cache directory writable.
RUN mkdir -p /tmp/huggingface && chmod -R 777 /tmp/huggingface

EXPOSE 7860

# One worker: the model is loaded per process, and a free CPU host has room for
# exactly one copy. Threads let health checks answer during a generation.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "180", "--access-logfile", "-", "main:app"]
