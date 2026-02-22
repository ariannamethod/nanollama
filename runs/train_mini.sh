#!/bin/bash
# Train nanollama mini (150M) — current WTForacle scale (~3h per training)
# Usage: bash runs/train_mini.sh [personality.jsonl]
set -e
PERSONALITY_FILE="${1:-}"
echo "nanollama — Mini (150M, depth=16, EN-only)"
if [ -n "$PERSONALITY_FILE" ]; then
    exec bash runs/lambda_train.sh --name mini --personality "$PERSONALITY_FILE" --save-every 1000
else
    exec bash runs/lambda_train.sh --name mini --base-only --save-every 1000
fi
