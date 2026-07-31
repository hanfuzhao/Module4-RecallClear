FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=4 \
    PORT=7860

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY scripts/bake_model.py /tmp/bake_model.py
RUN python /tmp/bake_model.py

COPY main.py ./
COPY scripts ./scripts
COPY templates ./templates
COPY static ./static
COPY models/adapter ./models/adapter
COPY data ./data

RUN chmod -R a+rwX /opt/hf

EXPOSE 7860

CMD exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 \
    --timeout 300 --access-logfile - main:app
