# Step 11: container for the tagging API, with the champion model baked in.
# Build after `python src/export_champion.py` has written ./serving_model.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/app/serving_model \
    REQUEST_LOG=/app/logs/requests.jsonl

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install exactly the library versions the model was registered with (MLflow wrote them),
# swapping full MLflow for the much smaller mlflow-skinny, plus the web server.
COPY serving_model/requirements.txt /tmp/model-requirements.txt
RUN MLFLOW_VERSION=$(grep -E '^mlflow==' /tmp/model-requirements.txt | cut -d= -f3) \
    && grep -vE '^mlflow==' /tmp/model-requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
       "mlflow-skinny==${MLFLOW_VERSION}" fastapi uvicorn python-multipart pyyaml

# Code and settings first, the big model folder last.
COPY configs/ configs/
COPY src/serve/ serve/
COPY serving_model/ serving_model/

# Run as a normal user, not root.
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app/logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]