#!/bin/bash
# =============================================================================
# Train nanollama large (3.7B) on Lambda 8×A100
#
# Tier 3 multilingual: all 13 languages (96K vocab)
# Requires Tier 3 tokenizer:
#   python -m scripts.train_tokenizer --tier 3
#
# Hardware: 8×A100 80GB (~96 hours per training)
# Usage: bash runs/train_large.sh [personality.jsonl]
# =============================================================================
set -e
PERSONALITY_FILE="${1:-}"

TOKENIZER_DIR="$HOME/.cache/nanollama/tokenizer_tier3"
if [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
    echo "ERROR: Tier 3 tokenizer not found. Train it first:"
    echo "  python -m scripts.train_tokenizer --tier 3"
    exit 1
fi

echo "nanollama — Large (3.7B, depth=32, Tier 3: 13 languages, 96K vocab)"
if [ -n "$PERSONALITY_FILE" ]; then
    exec bash runs/lambda_train.sh --name large --personality "$PERSONALITY_FILE" --save-every 5000
else
    exec bash runs/lambda_train.sh --name large --base-only --save-every 5000
fi
