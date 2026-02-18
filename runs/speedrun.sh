#!/bin/bash
# nanollama speedrun - train GPT-2 equivalent model in ~3 hours on 8xA100

set -e

echo "========================================"
echo "  nanollama Speedrun"
echo "  Train Llama 3 model from scratch"
echo "========================================"

# Configuration
DEPTH=24
RUN_NAME="speedrun_d${DEPTH}"
MODEL_TAG="speedrun"

# Multi-GPU training
echo "Starting training with depth=$DEPTH..."
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=$DEPTH \
    --run="$RUN_NAME" \
    --model-tag="$MODEL_TAG" \
    --total-batch-size=1048576 \
    --num-iterations=10000 \
    --wandb

# Evaluate
echo "Evaluating base model..."
python -m scripts.base_eval --model-tag="$MODEL_TAG"

# Start chat interface
echo "Starting chat web UI..."
echo "Open http://localhost:8000 to talk to your model!"
python -m scripts.chat_web --model-tag="$MODEL_TAG"
