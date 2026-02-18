#!/bin/bash
# nanollama miniseries - train all model sizes

set -e

echo "========================================"
echo "  nanollama Miniseries"
echo "  Train models at various scales"
echo "========================================"

# Model series: nano -> micro -> mini -> small -> medium -> large
DEPTHS=(6 12 16 24 32)
NAMES=("nano" "micro" "mini" "small" "medium")

for i in "${!DEPTHS[@]}"; do
    DEPTH=${DEPTHS[$i]}
    NAME=${NAMES[$i]}
    
    echo ""
    echo "========================================"
    echo "  Training $NAME model (depth=$DEPTH)"
    echo "========================================"
    
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
        --depth=$DEPTH \
        --run="miniseries_${NAME}" \
        --model-tag="${NAME}" \
        --wandb
    
    # Evaluate
    python -m scripts.base_eval --model-tag="${NAME}" --output="eval_${NAME}.json"
done

echo ""
echo "========================================"
echo "  Miniseries complete!"
echo "========================================"
