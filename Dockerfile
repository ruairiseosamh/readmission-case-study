# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src ./src
COPY app ./app
COPY data ./data

# Default model path; you can override with -e MODEL_PATH
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/artifacts/model.joblib

# Train on container build (optional). Comment out to skip training during build.
# RUN python -m readmission.train

EXPOSE 8000
# Cloud Run provides PORT env var; default to 8000
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
