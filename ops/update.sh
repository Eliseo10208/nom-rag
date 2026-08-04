#!/bin/sh
set -eu

git pull --ff-only
docker compose build --pull
docker compose up -d --wait --wait-timeout 240
docker compose ps
