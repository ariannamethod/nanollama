#!/bin/bash
# Lambda Cloud setup for nanollama
# One-command setup for Lambda GPU instances (A100/H100)

set -e

echo "========================================"
echo "  nanollama Lambda Cloud Setup"
echo "========================================"
# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name --format=csv,noheader

# Install uv if not present (fast Python package manager)
if ! command -v uv &> /dev/null; then
    echo "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Clone repo if needed
if [ ! -d "nanollama" ]; then
    echo "Cloning nanollama..."
    git clone https://github.com/ariannamethod/nanollama.git
    cd nanollama
else
    cd nanollama
    git pull
fi

# Create virtual environment and install dependencies
echo ""
echo "Setting up Python environment..."
uv venv
source .venv/bin/activate

# Install PyTorch >= 2.4.0 with CUDA 12.x
echo "Installing PyTorch >= 2.4.0 with CUDA 12.x..."
uv pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install minimal dependencies (NO HuggingFace libraries!)
# We don't need transformers, accelerate, datasets etc.
echo "Installing minimal dependencies..."
uv pip install sentencepiece numpy tqdm

# Install optional dependencies
uv pip install wandb  # For logging (optional)

# Install nanollama in editable mode
uv pip install -e .

# Configure NCCL for multi-GPU training
echo ""
echo "Configuring NCCL for multi-GPU..."
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=1  # Required for some Lambda configs

# Add to .bashrc for persistence (only if not already present)
if ! grep -q "NCCL configuration for nanollama" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'NCCL_CONFIG'
# NCCL configuration for nanollama
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=1
NCCL_CONFIG
fi

# Create data directory
mkdir -p ~/.cache/nanollama/data

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import sentencepiece; print('SentencePiece: OK')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "========================================"
echo "  ✅ Setup complete!"
echo ""
echo "  Quick test (CPU, ~2 min):"
echo "    python -m tests.smoke_test"
echo ""
echo "  Full training run:"
echo "    bash runs/lambda_train.sh --name nano --base-only"
echo ""
echo "  Prepare FineWeb-Edu data:"
echo "    python -m data.prepare_fineweb --num-samples 200000"
echo "========================================"
