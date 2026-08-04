#!/bin/sh
set -eu

chroma_dir="${NOMRAG_CHROMA_DIR:-/var/lib/nom-rag/chroma}"
traces_dir="${NOMRAG_TRACES_DIR:-/var/lib/nom-rag/traces}"
seed_dir="/opt/nom-rag/seed-chroma"

mkdir -p "$chroma_dir" "$traces_dir"

if [ ! -f "$chroma_dir/chroma.sqlite3" ]; then
    echo "Initializing persistent Chroma volume from the image snapshot..."
    cp -a "$seed_dir/." "$chroma_dir/"
fi

exec "$@"
