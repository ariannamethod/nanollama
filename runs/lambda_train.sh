#!/bin/bash
# Universal nanollama training script for Lambda Cloud
#
# One script to train any model size with optional personality injection.
# Handles setup, data prep, training, gamma extraction, and GGUF export.
#
# Usage:
#   bash runs/lambda_train.sh --name mini                          # base only
#   bash runs/lambda_train.sh --name mini --personality data.jsonl  # base + personality + gamma
#   bash runs/lambda_train.sh --name micro --steps 3000             # override steps
#
# The --name parameter selects from predefined configs:
#   nano(34M), micro(69M), mini(150M), small(336M), medium(1.6B), large(3.7B)

set -e

# ============================================================
# Config tables — one row per model size
# Columns: DEPTH, STEPS, BATCH, SAMPLES, GPU_NOTE
# ============================================================
declare -A CFG_DEPTH=( [nano]=6 [micro]=12 [mini]=16 [small]=24 [medium]=28 [large]=32 )
declare -A CFG_STEPS=( [nano]=1000 [micro]=5000 [mini]=10000 [small]=20000 [medium]=50000 [large]=100000 )
declare -A CFG_BATCH=( [nano]=65536 [micro]=131072 [mini]=262144 [small]=524288 [medium]=1048576 [large]=2097152 )
declare -A CFG_SAMPLES=( [nano]=200000 [micro]=500000 [mini]=1000000 [small]=3000000 [medium]=10000000 [large]=10000000 )

# ============================================================
# Parse arguments
# ============================================================
NAME=""
PERSONALITY_FILE=""
PERSONALITY_RATIO=0.20
STEPS_OVERRIDE=""
SAMPLES_OVERRIDE=""
TAG_SUFFIX=""
USE_WANDB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)       NAME="$2"; shift 2 ;;
        --personality) PERSONALITY_FILE="$2"; shift 2 ;;
        --ratio)      PERSONALITY_RATIO="$2"; shift 2 ;;
        --steps)      STEPS_OVERRIDE="$2"; shift 2 ;;
        --samples)    SAMPLES_OVERRIDE="$2"; shift 2 ;;
        --tag)        TAG_SUFFIX="$2"; shift 2 ;;
        --wandb)      USE_WANDB=true; shift ;;
        --help|-h)
            echo "Usage: bash runs/lambda_train.sh --name <size> [options]"
            echo ""
            echo "Required:"
            echo "  --name <size>        Model size: nano, micro, mini, small, medium, large"
            echo ""
            echo "Optional:"
            echo "  --personality <file>  JSONL file for personality training + gamma extraction"
            echo "  --ratio <0.0-1.0>    Personality data ratio (default: 0.20)"
            echo "  --steps <N>          Override training steps"
            echo "  --samples <N>        Override num FineWeb samples for data prep"
            echo "  --tag <suffix>       Custom tag suffix (default: model name)"
            echo "  --wandb              Enable wandb logging (must be logged in)"
            echo ""
            echo "Examples:"
            echo "  bash runs/lambda_train.sh --name mini"
            echo "  bash runs/lambda_train.sh --name mini --personality wtforacle.jsonl"
            echo "  bash runs/lambda_train.sh --name micro --steps 3000 --samples 300000"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$NAME" ]; then
    echo "ERROR: --name is required. Use --help for usage."
    exit 1
fi

if [ -z "${CFG_DEPTH[$NAME]}" ]; then
    echo "ERROR: Unknown model '$NAME'. Available: nano, micro, mini, small, medium, large"
    exit 1
fi

# Resolve config
DEPTH=${CFG_DEPTH[$NAME]}
NUM_STEPS=${STEPS_OVERRIDE:-${CFG_STEPS[$NAME]}}
BATCH_SIZE=${CFG_BATCH[$NAME]}
NUM_SAMPLES=${SAMPLES_OVERRIDE:-${CFG_SAMPLES[$NAME]}}
TAG=${TAG_SUFFIX:-$NAME}

WANDB_FLAG=""
if $USE_WANDB; then
    WANDB_FLAG="--wandb"
fi

echo "========================================"
echo "  nanollama — ${NAME} training"
echo "  depth=$DEPTH, steps=$NUM_STEPS, batch=$BATCH_SIZE"
echo "  samples=$NUM_SAMPLES, personality_ratio=$PERSONALITY_RATIO"
if [ -n "$PERSONALITY_FILE" ]; then
    echo "  personality: $PERSONALITY_FILE"
fi
echo "========================================"

# ============================================================
# GPU detection
# ============================================================
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs: $NUM_GPUS × $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
    echo ""
    echo "WARNING: H100 detected — known driver bug (Error 802, Feb 2026)."
    echo "Training may fail. Use A100 instead."
    echo ""
fi

# ============================================================
# Navigate to repo root
# ============================================================
if [ -f "pyproject.toml" ] && [ -d "nanollama" ]; then
    echo "Already in nanollama repo"
elif [ -d "nanollama" ] && [ -f "nanollama/pyproject.toml" ]; then
    cd nanollama
else
    echo "Cloning nanollama..."
    git clone https://github.com/ariannamethod/nanollama.git
    cd nanollama
fi

echo "Working directory: $(pwd)"

# ============================================================
# Install deps
# ============================================================
pip install sentencepiece numpy tqdm datasets filelock 2>/dev/null
pip install . 2>/dev/null || true

# ============================================================
# Step 1: Prepare FineWeb-Edu data
# ============================================================
DATA_DIR="$HOME/.cache/nanollama/data/fineweb"
SHARD_COUNT=$(ls "$DATA_DIR"/*.bin 2>/dev/null | wc -l)

if [ "$SHARD_COUNT" -lt 3 ]; then
    echo ""
    echo "========================================"
    echo "  Step 1: Preparing FineWeb-Edu ($NUM_SAMPLES samples)"
    echo "========================================"
    python -u -m data.prepare_fineweb --num-samples $NUM_SAMPLES
else
    echo ""
    echo "Step 1: FineWeb-Edu data exists ($SHARD_COUNT shards), skipping"
fi

# ============================================================
# Step 2: Prepare personality data (if provided)
# ============================================================
PERSONALITY_DIR=""
PERSONALITY_ARGS=""

if [ -n "$PERSONALITY_FILE" ]; then
    PERSONALITY_DIR="$HOME/.cache/nanollama/data/personality"

    # Find personality file
    if [ -f "$PERSONALITY_FILE" ]; then
        PERSONALITY_INPUT="$PERSONALITY_FILE"
    elif [ -f "$HOME/$PERSONALITY_FILE" ]; then
        PERSONALITY_INPUT="$HOME/$PERSONALITY_FILE"
    else
        echo "ERROR: Cannot find personality file: $PERSONALITY_FILE"
        exit 1
    fi

    echo ""
    echo "========================================"
    echo "  Step 2: Preparing personality data"
    echo "========================================"
    python -u -m data.prepare_personality --input "$PERSONALITY_INPUT" --output-dir "$PERSONALITY_DIR"

    PERSONALITY_ARGS="--personality-dir=$PERSONALITY_DIR --personality-ratio=$PERSONALITY_RATIO"
fi

# ============================================================
# Training helper function
# ============================================================
run_training() {
    local RUN_TAG="$1"
    local EXTRA_ARGS="$2"
    local LOG_FILE="$3"

    echo ""
    echo "========================================"
    echo "  Training: $RUN_TAG"
    echo "  depth=$DEPTH, steps=$NUM_STEPS, batch=$BATCH_SIZE"
    echo "========================================"

    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "Multi-GPU: $NUM_GPUS GPUs"
        OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_train \
            --depth=$DEPTH \
            --run="$RUN_TAG" \
            --model-tag="$RUN_TAG" \
            --total-batch-size=$BATCH_SIZE \
            --num-iterations=$NUM_STEPS \
            --save-every=1000 \
            $EXTRA_ARGS \
            $WANDB_FLAG \
            2>&1 | tee "$LOG_FILE"
    else
        python -u -m scripts.base_train \
            --depth=$DEPTH \
            --run="$RUN_TAG" \
            --model-tag="$RUN_TAG" \
            --total-batch-size=$BATCH_SIZE \
            --num-iterations=$NUM_STEPS \
            --save-every=1000 \
            $EXTRA_ARGS \
            $WANDB_FLAG \
            2>&1 | tee "$LOG_FILE"
    fi
}

# ============================================================
# Step 3: Train
# ============================================================
BASE_DIR="$HOME/.cache/nanollama"
TOKENIZER="$BASE_DIR/tokenizer/tokenizer.model"

if [ -n "$PERSONALITY_FILE" ]; then
    # Two-pass: base + personality, then gamma extraction
    run_training "${TAG}_base" "--personality-ratio=0.0" "train_base.log"
    run_training "${TAG}_personality" "$PERSONALITY_ARGS" "train_personality.log"

    BASE_CKPT="$BASE_DIR/checkpoints/${TAG}_base/checkpoint.pt"
    PERSONALITY_CKPT="$BASE_DIR/checkpoints/${TAG}_personality/checkpoint.pt"

    # ============================================================
    # Step 4: Extract gamma
    # ============================================================
    echo ""
    echo "========================================"
    echo "  Extracting gamma"
    echo "========================================"
    mkdir -p weights
    GAMMA_OUTPUT="weights/gamma_${TAG}_d${DEPTH}.npz"

    python -u -m scripts.extract_gamma \
        --personality_ckpt "$PERSONALITY_CKPT" \
        --base_ckpt "$BASE_CKPT" \
        --output "$GAMMA_OUTPUT"

    # ============================================================
    # Step 5: Export GGUF
    # ============================================================
    echo ""
    echo "========================================"
    echo "  Exporting to GGUF"
    echo "========================================"

    python -u -m scripts.export_gguf \
        --checkpoint "$PERSONALITY_CKPT" \
        --tokenizer "$TOKENIZER" \
        --output "weights/${TAG}-personality-f16.gguf" \
        --dtype f16

    python -u -m scripts.export_gguf \
        --checkpoint "$BASE_CKPT" \
        --tokenizer "$TOKENIZER" \
        --output "weights/${TAG}-base-f16.gguf" \
        --dtype f16

    echo ""
    echo "========================================"
    echo "  DONE — ${NAME} with personality"
    echo "========================================"
    echo ""
    echo "Artifacts:"
    echo "  weights/${TAG}-personality-f16.gguf"
    echo "  weights/${TAG}-base-f16.gguf"
    echo "  weights/${GAMMA_OUTPUT}"
else
    # Single pass: base model only
    run_training "${TAG}" "" "train_base.log"

    CKPT="$BASE_DIR/checkpoints/${TAG}/checkpoint.pt"

    echo ""
    echo "========================================"
    echo "  Exporting to GGUF"
    echo "========================================"
    mkdir -p weights

    python -u -m scripts.export_gguf \
        --checkpoint "$CKPT" \
        --tokenizer "$TOKENIZER" \
        --output "weights/${TAG}-f16.gguf" \
        --dtype f16

    echo ""
    echo "========================================"
    echo "  DONE — ${NAME} base"
    echo "========================================"
    echo ""
    echo "Artifacts:"
    echo "  weights/${TAG}-f16.gguf"
fi

echo ""
echo "Download:"
echo "  scp ubuntu@\$(hostname -I | awk '{print \$1}'):~/nanollama/weights/* ."
