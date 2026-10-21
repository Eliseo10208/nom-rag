# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NOMRAG_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Install the CPU-only torch wheel first so pip never pulls a CUDA build.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .

# Bake the embedding model and a matching Chroma snapshot into the image. A new
# persistent volume is seeded from this snapshot on its first container start.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
      "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY data/chunks.jsonl ./data/chunks.jsonl
RUN mkdir -p /opt/nom-rag/seed-chroma \
    && NOMRAG_CHROMA_DIR=/opt/nom-rag/seed-chroma \
       NOMRAG_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
       python -m nom_rag.cli build-index

COPY docker/entrypoint.sh /usr/local/bin/nom-rag-entrypoint
COPY docker/run-server.sh /usr/local/bin/nom-rag-server
RUN chmod 0755 /usr/local/bin/nom-rag-entrypoint /usr/local/bin/nom-rag-server \
    && useradd --system --uid 10001 --create-home --home-dir /home/nomrag nomrag \
    && mkdir -p /var/lib/nom-rag/chroma /var/lib/nom-rag/traces \
    && chown -R nomrag:nomrag /var/lib/nom-rag

USER nomrag

EXPOSE 8017

ENTRYPOINT ["/usr/local/bin/nom-rag-entrypoint"]
CMD ["/usr/local/bin/nom-rag-server"]
