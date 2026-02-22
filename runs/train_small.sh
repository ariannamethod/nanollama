#!/bin/bash
# =============================================================================
# Train nanollama small (336M) on Lambda A100
#
# English-only base model. Last monolingual model in the series.
# Next step up (goldie 1.1B) introduces multilingual tokenizer.
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
#   bash runs/train_small.sh                              # base only
#   bash runs/train_small.sh wtforacle.jsonl              # with personality
#   scp runs/train_small.sh ubuntu@<ip>:~/nanollama/runs/
#
# Hardware: 1×A100 80GB (~18 hours per training, ~36h total with personality)
# =============================================================================

set -e

PERSONALITY_FILE="${1:-}"

echo "========================================================"
echo "  nanollama — Small (336M, depth=24, EN-only)"
echo "========================================================"

if [ -n "$PERSONALITY_FILE" ]; then
    echo "  Personality: $PERSONALITY_FILE"
    echo "  Pipeline: data → base → personality → gamma → GGUF"
    exec bash runs/lambda_train.sh \
        --name small \
        --personality "$PERSONALITY_FILE" \
        --ratio 0.20 \
        --save-every 2000
else
    echo "  Pipeline: data → base → GGUF"
    exec bash runs/lambda_train.sh \
        --name small \
        --base-only \
        --save-every 2000
fi
