#!/bin/sh
set -eu

backup_dir="${1:-./backups}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$backup_dir"
backup_dir="$(cd "$backup_dir" && pwd)"

echo "Stopping the application for a consistent Chroma snapshot..."
docker compose stop app
trap 'docker compose start app >/dev/null 2>&1 || true' EXIT INT TERM

docker run --rm \
    -v nom-rag_chroma-data:/source:ro \
    -v "$backup_dir:/backup" \
    alpine:3.22 \
    tar -C /source -czf "/backup/chroma-$timestamp.tar.gz" .

docker run --rm \
    -v nom-rag_traces-data:/source:ro \
    -v "$backup_dir:/backup" \
    alpine:3.22 \
    tar -C /source -czf "/backup/traces-$timestamp.tar.gz" .

docker compose start app
trap - EXIT INT TERM

echo "Backup written to $backup_dir"
