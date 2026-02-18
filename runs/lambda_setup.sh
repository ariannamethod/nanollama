#!/bin/bash
# Lambda Cloud setup for nanollama
# One-command setup for Lambda A100 instances
#
# IMPORTANT: Avoid H100 instances — known driver bug (Error 802) as of Feb 2026.
# Use A100 80GB for stable training.

set -e

echo "========================================"
echo "  nanollama Lambda Cloud Setup"
echo "========================================"
echo ""
echo "⚠️  IMPORTANT: Avoid H100 instances!"
echo "    Known driver bug (Error 802) as of Feb 2026."
echo "    Use A100 80GB instances instead."
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name --format=csv,noheader

# Check if H100 (warn user)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
    echo ""
    echo "❌ H100 DETECTED - Known driver bug (Error 802)!"
    echo "   Training may fail with cryptic CUDA errors."
    echo "   STRONGLY recommend using A100 80GB instead."
    echo ""
    read -p "Continue anyway at your own risk? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please use A100 instances."
        exit 1
    fi
fi

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
echo "    bash runs/speedrun.sh"
echo ""
echo "  Prepare TinyStories data:"
echo "    python -m data.prepare_tinystories"
echo "========================================"
