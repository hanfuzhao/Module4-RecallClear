# Container image for the deployed RecallClear app (Cloud Run / any Docker host).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=4 \
    PORT=7860

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Bake the base model into the image at build time. Downloading it lazily at
# first request made the app depend on Hugging Face being reachable -- and
# rate-limitable -- from the serving host's shared egress IPs, which is exactly
# what happened (429 on the very first cold start).
COPY scripts/bake_model.py /tmp/bake_model.py
RUN python /tmp/bake_model.py

COPY main.py ./
COPY scripts ./scripts
COPY templates ./templates
COPY static ./static
COPY models/adapter ./models/adapter
# data/ holds the demo notices and, when present, the evaluation summary the
# metrics panel reads. Copying the directory keeps the latter optional.
COPY data ./data

# Some hosts run the container as a non-root user; keep the cache readable and
# writable either way. /opt/hf is image content, not a tmpfs, so the weights
# baked above are guaranteed to be present at boot.
RUN chmod -R a+rwX /opt/hf

EXPOSE 7860

# One worker: the model is loaded per process and the host has room for exactly
# one copy. Threads let health checks answer during a generation. Shell form so
# $PORT is honoured -- Cloud Run injects it (8080); 7860 is the local default.
CMD exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 \
    --timeout 300 --access-logfile - main:app
