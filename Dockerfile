# Serving image. Training happens outside; this only scores.
FROM python:3.12-slim

# libgomp1 provides the OpenMP runtime LightGBM links against. Without it the
# import fails and the pipeline silently falls back to sklearn's booster --
# which would not match the model artefact trained with LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir lightgbm

# Baked-in model. For a real deployment, mount it or pull from a registry at
# startup so the image and the model can be versioned independently.
COPY artifacts/models/ ./artifacts/models/

ENV FRD_MODEL_PATH=/app/artifacts/models/detector.joblib \
    FRD_MODEL_VERSION=0.1.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "frd.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
