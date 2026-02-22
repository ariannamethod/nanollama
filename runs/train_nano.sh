#!/bin/bash
# Train nanollama nano (34M) — smallest model, fast iteration (~20 min)
# Usage: bash runs/train_nano.sh [personality.jsonl]
set -e
PERSONALITY_FILE="${1:-}"
echo "nanollama — Nano (34M, depth=6, EN-only)"
if [ -n "$PERSONALITY_FILE" ]; then
    exec bash runs/lambda_train.sh --name nano --personality "$PERSONALITY_FILE"
else
    exec bash runs/lambda_train.sh --name nano --base-only
fi
