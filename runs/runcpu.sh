#!/bin/bash
# Run nanollama on CPU/MPS for testing
# Use tiny model for quick iteration

set -e

echo "========================================"
echo "  nanollama CPU/MPS Mode"
echo "  For testing and development"
echo "========================================"

# Tiny model for CPU
python -m scripts.base_train \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=2 \
    --total-batch-size=64 \
    --num-iterations=100 \
    --log-every=10 \
    --save-every=-1 \
    --eval-every=-1 \
    --sample-every=-1 \
    --core-metric-every=-1 \
    --model-tag="cpu_test"

echo ""
echo "CPU test complete!"
