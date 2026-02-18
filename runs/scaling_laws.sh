#!/bin/bash
# nanollama scaling laws experiments

set -e

echo "========================================"
echo "  nanollama Scaling Laws"
echo "  Sweep across model sizes and tokens"
echo "========================================"

# Small depths for quick scaling experiments
DEPTHS=(4 6 8 10 12 14 16)

for DEPTH in "${DEPTHS[@]}"; do
    echo ""
    echo "Training depth=$DEPTH..."
    
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
        --depth=$DEPTH \
        --run="scaling_d${DEPTH}" \
        --model-tag="scaling_d${DEPTH}" \
        --num-iterations=2000 \
        --core-metric-every=500 \
        --save-every=-1 \
        --sample-every=-1 \
        --wandb
done

echo ""
echo "Scaling experiments complete!"
echo "Analyze results in wandb."
