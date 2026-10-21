#!/bin/sh
set -eu

# Vercel and Cloud Run inject PORT. The VPS image keeps its existing 8017
# default when no platform-provided port is present.
port="${PORT:-8017}"

exec python -m uvicorn nom_rag.api:app \
    --host 0.0.0.0 \
    --port "$port" \
    --workers 1 \
    --limit-concurrency 32 \
    --timeout-keep-alive 10 \
    --proxy-headers \
    '--forwarded-allow-ips=*'
