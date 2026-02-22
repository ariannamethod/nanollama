#!/bin/bash
# =============================================================================
# Train nanollama goldie (1.1B) on Lambda A100
#
# FIRST MULTILINGUAL MODEL. Tier 1: English, Russian, French, German.
# Requires Tier 1 tokenizer (48K vocab) — train it first:
#   python -m scripts.train_tokenizer --tier 1
#
# Full pipeline:
#   1. Prepare multi-corpus data (FineWeb 55%, DCLM 25%, Stack 10%, Math 10%)
#   2. Prepare personality shard (if provided)
#   3. Train BASE model → for gamma extraction
#   4. Train PERSONALITY model (20% personality mix)
#   5. Extract gamma (personality - base)
#   6. Export both to GGUF
#
# Usage:
#   bash runs/train_goldie.sh                              # base only
#   bash runs/train_goldie.sh wtforacle.jsonl              # with personality
#   scp runs/train_goldie.sh ubuntu@<ip>:~/nanollama/runs/
#
# Hardware: 1-2×A100 80GB (~24 hours per training, ~48h total with personality)
#
# IMPORTANT: Train the Tier 1 tokenizer BEFORE running this script:
#   python -m scripts.train_tokenizer --tier 1
#   Then reference it with --tokenizer-dir ~/.cache/nanollama/tokenizer_tier1
# =============================================================================

set -e

PERSONALITY_FILE="${1:-}"

# Check for Tier 1 tokenizer
TOKENIZER_DIR="$HOME/.cache/nanollama/tokenizer_tier1"
if [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
    echo "ERROR: Tier 1 multilingual tokenizer not found at $TOKENIZER_DIR"
    echo ""
    echo "Train it first:"
    echo "  python -m scripts.train_tokenizer --tier 1"
    echo ""
    echo "This takes ~10-20 minutes (downloads CulturaX + FineWeb-Edu samples)."
    exit 1
fi

echo "========================================================"
echo "  nanollama — Goldie (1.1B, depth=20, Tier 1: EN/RU/FR/DE)"
echo "  Tokenizer: $TOKENIZER_DIR (48K vocab)"
echo "========================================================"

if [ -n "$PERSONALITY_FILE" ]; then
    echo "  Personality: $PERSONALITY_FILE"
    echo "  Pipeline: data → base → personality → gamma → GGUF"
    exec bash runs/lambda_train.sh \
        --name goldie \
        --personality "$PERSONALITY_FILE" \
        --ratio 0.20 \
        --save-every 2000
else
    echo "  Pipeline: data → base → GGUF"
    exec bash runs/lambda_train.sh \
        --name goldie \
        --base-only \
        --save-every 2000
fi
