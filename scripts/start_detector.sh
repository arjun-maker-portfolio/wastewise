#!/bin/bash

set -e

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs data

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

sleep 5

exec python3 scripts/app.py \
    --camera 0 \
    --interval 2 \
    --use-firebase \
    --log-file "$PROJECT_DIR/logs/detector.log"
