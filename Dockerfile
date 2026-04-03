# DataWranglerEnv — Dockerfile for Hugging Face Spaces
#
# Build: docker build -t data-wrangler-env .
# Run:   docker run -p 7860:7860 data-wrangler-env

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY data_wrangler_env/ /app/data_wrangler_env/
COPY inference.py /app/inference.py

# Install using pyproject.toml from the package dir
WORKDIR /app/data_wrangler_env
RUN pip install --no-cache-dir -e .

# Work from /app so that `data_wrangler_env` is a proper package on the path
WORKDIR /app

# PYTHONPATH: /app is already on path since we're the WORKDIR
ENV PYTHONPATH="/app"

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run: uvicorn data_wrangler_env.server.app:app from /app
# This makes Python see data_wrangler_env as a package, so relative imports work
CMD ["python", "-m", "uvicorn", "data_wrangler_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
