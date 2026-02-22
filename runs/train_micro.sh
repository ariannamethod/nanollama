#!/bin/bash
# Train nanollama micro (69M) — micro-Yent scale (~40 min)
# Usage: bash runs/train_micro.sh [personality.jsonl]
set -e
PERSONALITY_FILE="${1:-}"
echo "nanollama — Micro (69M, depth=12, EN-only)"
if [ -n "$PERSONALITY_FILE" ]; then
    exec bash runs/lambda_train.sh --name micro --personality "$PERSONALITY_FILE"
else
    exec bash runs/lambda_train.sh --name micro --base-only
fi
