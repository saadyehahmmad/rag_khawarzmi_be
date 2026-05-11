# --------------------------------------------------------------------------- #
# Al-Khwarizmi RAG API — multi-stage Dockerfile
#
# Stage 1 (builder): install Python dependencies into /install
# Stage 2 (runtime): copy only the installed packages + app code — no build tools
#
# Build:
#   docker build -t alkhwarizmi-rag:latest .
#
# Run (with .env file):
#   docker run --env-file .env -p 8000:8000 alkhwarizmi-rag:latest
#
# Run with GPU (requires nvidia-docker2):
#   docker run --gpus all --env-file .env -p 8000:8000 alkhwarizmi-rag:latest
# --------------------------------------------------------------------------- #

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to build wheels (chromadb, tokenizers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so workers never fetch at runtime.
# Store in /opt/hf-cache so it can be copied to a world-readable location.
COPY agent/env_utils.py /tmp/env_utils.py
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    HF_HOME=/opt/hf-cache \
    python -c "
import os, sys
sys.path.insert(0, '/install/lib/python3.11/site-packages')
model = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large')
from sentence_transformers import SentenceTransformer
SentenceTransformer(model)
print('Embedding model cached:', model)
"


# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security.
RUN addgroup --system rag && adduser --system --ingroup rag rag

WORKDIR /app

# Copy installed packages from builder.
COPY --from=builder /install /usr/local
# Copy the pre-downloaded embedding model cache (world-readable, owned by root is fine — read-only access works).
COPY --from=builder /opt/hf-cache /opt/hf-cache

# Tell HuggingFace / sentence-transformers where the cache lives.
ENV HF_HOME=/opt/hf-cache

# Copy application source.
COPY agent/       agent/
COPY api/         api/
COPY ingestion/   ingestion/
COPY eval/        eval/
COPY start.py     start.py

# Runtime directories owned by the non-root user.
RUN mkdir -p vector_stores logs memory \
 && chown -R rag:rag /app

USER rag

# Expose API port (configure via API_PORT env var — default 8000).
EXPOSE 8000

# Health check: liveness probe (light — does not load embeddings).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Use start.py so $PORT is respected and workers=1 avoids OOM on low-memory hosts.
# For multi-worker production hosts set UVICORN_WORKERS env var and override CMD.
CMD ["python", "start.py", "--no-reload"]
