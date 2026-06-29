FROM python:3.11-slim

WORKDIR /app

# System deps — only what's needed to compile any C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached until pyproject.toml changes)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[server]" || pip install --no-cache-dir \
        fastapi>=0.100.0 \
        uvicorn>=0.20.0 \
        httpx>=0.24.0 \
        pydantic>=2.0.0 \
        cryptography>=41.0.0 \
        python-multipart>=0.0.6 \
        anthropic>=0.40.0

# Copy source
COPY pact_ax/ ./pact_ax/

# Data volume — all SQLite DBs land here
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "pact_ax.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
