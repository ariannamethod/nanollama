#!/bin/bash
# =============================================================================
# Train nanollama medium (1.6B) on Lambda multi-GPU
#
# Tier 2 multilingual: EN, RU, FR, DE + ES, PT, UK, TR (64K vocab)
# Requires Tier 2 tokenizer:
#   python -m scripts.train_tokenizer --tier 2
#
# Hardware: 4×A100 80GB (~48 hours per training)
# Usage: bash runs/train_medium.sh [personality.jsonl]
# =============================================================================
set -e
PERSONALITY_FILE="${1:-}"

TOKENIZER_DIR="$HOME/.cache/nanollama/tokenizer_tier2"
if [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
    echo "ERROR: Tier 2 tokenizer not found. Train it first:"
    echo "  python -m scripts.train_tokenizer --tier 2"
    exit 1
fi

echo "nanollama — Medium (1.6B, depth=32, Tier 2: 8 languages, 64K vocab)"
if [ -n "$PERSONALITY_FILE" ]; then
    exec bash runs/lambda_train.sh --name medium --personality "$PERSONALITY_FILE" --save-every 5000
else
    exec bash runs/lambda_train.sh --name medium --base-only --save-every 5000
fi
