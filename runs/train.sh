#!/bin/bash
# =============================================================================
# nanollama — Main Training Entry Point
#
# "For idiots" mode: pick a size, optionally pass personality file, done.
# Each size has its own script that wraps lambda_train.sh.
#
# Usage:
#   bash runs/train.sh                     # interactive menu
#   bash runs/train.sh mini                # train mini, base only
#   bash runs/train.sh mini data.jsonl     # train mini with personality
#   bash runs/train.sh goldie data.jsonl   # train goldie (needs Tier 1 tokenizer)
#
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIZE="${1:-}"
PERSONALITY="${2:-}"

if [ -z "$SIZE" ]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║                    nanollama trainer                        ║"
    echo "  ╠══════════════════════════════════════════════════════════════╣"
    echo "  ║                                                            ║"
    echo "  ║   1) nano     34M   ~20 min   1×GPU   EN                   ║"
    echo "  ║   2) micro    69M   ~40 min   1×GPU   EN                   ║"
    echo "  ║   3) mini    150M   ~3 hrs    1×GPU   EN                   ║"
    echo "  ║   4) small   336M   ~18 hrs   1×A100  EN                   ║"
    echo "  ║   5) goldie  1.1B   ~24 hrs   1-2×A100 80GB  EN/RU/FR/DE  ║"
    echo "  ║   6) medium  1.6B   ~48 hrs   4×A100 80GB    8 languages   ║"
    echo "  ║   7) large   3.7B   ~96 hrs   8×A100 80GB    13 languages  ║"
    echo "  ║   8) big     7.0B   ~200 hrs  8×A100 80GB    13 languages  ║"
    echo "  ║                                                            ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
    read -p "  Pick a size (name or number): " CHOICE

    case "$CHOICE" in
        1|nano)    SIZE="nano" ;;
        2|micro)   SIZE="micro" ;;
        3|mini)    SIZE="mini" ;;
        4|small)   SIZE="small" ;;
        5|goldie)  SIZE="goldie" ;;
        6|medium)  SIZE="medium" ;;
        7|large)   SIZE="large" ;;
        8|big)     SIZE="big" ;;
        *) echo "Unknown: $CHOICE"; exit 1 ;;
    esac

    # Ask for personality file
    echo ""
    read -p "  Personality JSONL file (or press Enter for base-only): " PERSONALITY
    echo ""
fi

# Validate
TRAIN_SCRIPT="$SCRIPT_DIR/train_${SIZE}.sh"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: No training script for '$SIZE'"
    echo "Available: nano, micro, mini, small, goldie, medium, large"
    exit 1
fi

echo "Launching: bash runs/train_${SIZE}.sh ${PERSONALITY}"
echo ""

exec bash "$TRAIN_SCRIPT" "$PERSONALITY"
