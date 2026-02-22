#!/bin/bash
# =============================================================================
# Train nanollama big (7.0B) on Lambda multi-GPU     *** PLANNED ***
#
# Tier 3 multilingual: all 13 languages (96K vocab)
# Flagship model — comparable to Llama 3 8B, Mistral 7B.
#
# Requires Tier 3 tokenizer:
#   python -m scripts.train_tokenizer --tier 3
#
# Hardware: 8×A100 80GB (~200 hours) or 8×H100 (~80 hours)
# Data: 20B tokens multi-corpus, ~40 GB on disk
# Usage: bash runs/train_big.sh [personality.jsonl]
# =============================================================================
set -e
PERSONALITY_FILE="${1:-}"

TOKENIZER_DIR="$HOME/.cache/nanollama/tokenizer_tier3"
if [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
    echo "ERROR: Tier 3 tokenizer not found. Train it first:"
    echo "  python -m scripts.train_tokenizer --tier 3"
    exit 1
fi

echo "========================================================"
echo "  nanollama — Big (7.0B, depth=36, Tier 3: 13 languages, 96K vocab)"
echo "  Tokenizer: $TOKENIZER_DIR"
echo "  Data: 20B tokens multi-corpus (~40 GB)"
echo "  GPU: 8×A100 80GB recommended"
echo "========================================================"

if [ -n "$PERSONALITY_FILE" ]; then
    echo "  Personality: $PERSONALITY_FILE"
    exec bash runs/lambda_train.sh --name big --personality "$PERSONALITY_FILE" --save-every 5000
else
    exec bash runs/lambda_train.sh --name big --base-only --save-every 5000
fi
