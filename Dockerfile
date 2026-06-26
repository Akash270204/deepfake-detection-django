# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --target /install

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Minimal runtime libs for OpenCV / TF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.10/site-packages

# Copy project source (excludes whatever is in .dockerignore)
COPY . .

# Create required runtime directories
RUN mkdir -p logs media staticfiles

# Collect static files
RUN DJANGO_SETTINGS_MODULE=deepfake_detection.settings \
    python manage.py collectstatic --noinput

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Gunicorn: 2 workers (memory-aware for TF models), timeout 300 s for ML inference
CMD ["gunicorn", "deepfake_detection.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "300", \
     "--worker-class", "sync", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
