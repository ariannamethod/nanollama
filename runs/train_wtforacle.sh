#!/bin/bash
# Train nanollama mini (150M) with WTForacle personality on Lambda A100
#
# Full pipeline:
#   1. Prepare FineWeb-Edu data shards
#   2. Prepare WTForacle personality shard
#   3. Train BASE model (no personality) — for gamma extraction
#   4. Train PERSONALITY model (20% wtforacle)
#   5. Extract gamma (personality - base)
#   6. Export both to GGUF
#
# Usage:
#   scp this script + wtforacle_identity_v3_final.jsonl to Lambda
#   bash train_wtforacle.sh
#
# Expected time: ~6 hours total on 1xA100 (~3h per training)
#   With 8xA100: ~1.5h total (~45 min per training)

set -e

DEPTH=16
NUM_STEPS=10000
PERSONALITY_RATIO=0.20
PERSONALITY_FILE="wtforacle_identity_v3_final.jsonl"
RUN_PREFIX="wtforacle_mini"

echo "========================================"
echo "  nanollama — WTForacle Mini (150M)"
echo "  depth=$DEPTH, steps=$NUM_STEPS"
echo "========================================"

# Detect GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs detected: $NUM_GPUS"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Check for H100 (known broken)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
    echo ""
    echo "WARNING: H100 detected — known driver bug (Error 802)."
    echo "Training may fail. Use A100 instead."
    echo ""
fi

# Navigate to repo root
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

# Install deps if needed
pip install sentencepiece numpy tqdm datasets 2>/dev/null
pip install . 2>/dev/null || true

# ========================================
# Step 1: Prepare FineWeb-Edu data
# ========================================
DATA_DIR="$HOME/.cache/nanollama/data/fineweb"
if [ ! -d "$DATA_DIR" ] || [ $(ls "$DATA_DIR"/*.bin 2>/dev/null | wc -l) -lt 5 ]; then
    echo ""
    echo "========================================"
    echo "  Step 1: Preparing FineWeb-Edu data"
    echo "========================================"
    python -u -m data.prepare_fineweb --num-shards 22
else
    echo ""
    echo "Step 1: FineWeb-Edu data already exists ($(ls "$DATA_DIR"/*.bin | wc -l) shards)"
fi

# ========================================
# Step 2: Prepare WTForacle personality data
# ========================================
PERSONALITY_DIR="$HOME/.cache/nanollama/data/personality"
echo ""
echo "========================================"
echo "  Step 2: Preparing WTForacle personality data"
echo "========================================"

# Find the personality file
if [ -f "$PERSONALITY_FILE" ]; then
    PERSONALITY_INPUT="$PERSONALITY_FILE"
elif [ -f "data/$PERSONALITY_FILE" ]; then
    PERSONALITY_INPUT="data/$PERSONALITY_FILE"
elif [ -f "$HOME/$PERSONALITY_FILE" ]; then
    PERSONALITY_INPUT="$HOME/$PERSONALITY_FILE"
else
    echo "ERROR: Cannot find $PERSONALITY_FILE"
    echo "Upload it to Lambda first:"
    echo "  scp wtforacle_identity_v3_final.jsonl ubuntu@<lambda-ip>:~/"
    exit 1
fi

echo "Using personality file: $PERSONALITY_INPUT"
python -u -m data.prepare_personality --input "$PERSONALITY_INPUT" --output-dir "$PERSONALITY_DIR"

# ========================================
# Step 3: Train BASE model (no personality)
# ========================================
echo ""
echo "========================================"
echo "  Step 3: Training BASE model (no personality)"
echo "  depth=$DEPTH, steps=$NUM_STEPS"
echo "========================================"

BASE_TAG="${RUN_PREFIX}_base"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU training with $NUM_GPUS GPUs"
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_train \
        --depth=$DEPTH \
        --run="${BASE_TAG}" \
        --model-tag="${BASE_TAG}" \
        --personality-ratio=0.0 \
        --num-iterations=$NUM_STEPS \
        --save-every=1000 \
        --wandb \
        2>&1 | tee train_base.log
else
    echo "Single GPU training"
    python -u -m scripts.base_train \
        --depth=$DEPTH \
        --run="${BASE_TAG}" \
        --model-tag="${BASE_TAG}" \
        --personality-ratio=0.0 \
        --num-iterations=$NUM_STEPS \
        --save-every=1000 \
        --wandb \
        2>&1 | tee train_base.log
fi

echo "Base training complete."

# ========================================
# Step 4: Train PERSONALITY model (20% wtforacle)
# ========================================
echo ""
echo "========================================"
echo "  Step 4: Training PERSONALITY model (${PERSONALITY_RATIO} wtforacle)"
echo "  depth=$DEPTH, steps=$NUM_STEPS"
echo "========================================"

PERSONALITY_TAG="${RUN_PREFIX}_personality"

if [ "$NUM_GPUS" -gt 1 ]; then
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_train \
        --depth=$DEPTH \
        --run="${PERSONALITY_TAG}" \
        --model-tag="${PERSONALITY_TAG}" \
        --personality-dir="$PERSONALITY_DIR" \
        --personality-ratio=$PERSONALITY_RATIO \
        --num-iterations=$NUM_STEPS \
        --save-every=1000 \
        --wandb \
        2>&1 | tee train_personality.log
else
    python -u -m scripts.base_train \
        --depth=$DEPTH \
        --run="${PERSONALITY_TAG}" \
        --model-tag="${PERSONALITY_TAG}" \
        --personality-dir="$PERSONALITY_DIR" \
        --personality-ratio=$PERSONALITY_RATIO \
        --num-iterations=$NUM_STEPS \
        --save-every=1000 \
        --wandb \
        2>&1 | tee train_personality.log
fi

echo "Personality training complete."

# ========================================
# Step 5: Extract gamma
# ========================================
echo ""
echo "========================================"
echo "  Step 5: Extracting gamma"
echo "========================================"

BASE_DIR="$HOME/.cache/nanollama"
BASE_CKPT="$BASE_DIR/checkpoints/${BASE_TAG}/checkpoint.pt"
PERSONALITY_CKPT="$BASE_DIR/checkpoints/${PERSONALITY_TAG}/checkpoint.pt"
GAMMA_OUTPUT="weights/gamma_wtforacle_d${DEPTH}.npz"

mkdir -p weights

python -u scripts/extract_gamma.py \
    --personality_ckpt "$PERSONALITY_CKPT" \
    --base_ckpt "$BASE_CKPT" \
    --output "$GAMMA_OUTPUT"

echo "Gamma saved: $GAMMA_OUTPUT"

# ========================================
# Step 6: Export to GGUF
# ========================================
echo ""
echo "========================================"
echo "  Step 6: Exporting to GGUF"
echo "========================================"

TOKENIZER="$BASE_DIR/tokenizer/tokenizer.model"

# Export personality model (main model)
python -u scripts/export_gguf.py \
    --checkpoint "$PERSONALITY_CKPT" \
    --tokenizer "$TOKENIZER" \
    --output "weights/wtforacle-mini-f16.gguf" \
    --dtype f16

# Export base model too (for comparison)
python -u scripts/export_gguf.py \
    --checkpoint "$BASE_CKPT" \
    --tokenizer "$TOKENIZER" \
    --output "weights/mini-base-f16.gguf" \
    --dtype f16

echo ""
echo "========================================"
echo "  DONE!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  weights/wtforacle-mini-f16.gguf    — personality model"
echo "  weights/mini-base-f16.gguf         — base model"
echo "  weights/gamma_wtforacle_d${DEPTH}.npz — personality vector"
echo ""
echo "Download to Mac:"
echo "  scp ubuntu@\$(hostname -I | awk '{print \$1}'):~/nanollama/weights/* ."
echo ""
echo "Test with Go engine:"
echo "  ./nanollama --model weights/wtforacle-mini-f16.gguf --interactive"
echo "  ./nanollama --model weights/mini-base-f16.gguf --gamma weights/gamma_wtforacle_d${DEPTH}.npz --interactive"
