#!/bin/bash
# Lambda Cloud setup for nanollama
# One-command setup for Lambda A100 instances

set -e

echo "========================================"
echo "  nanollama Lambda Cloud Setup"
echo "========================================"

# Known issue: H100 has driver bug (Error 802) as of Feb 2026
# Use A100 instances instead

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name --format=csv,noheader

# Check if H100 (warn user)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
    echo ""
    echo "WARNING: H100 detected. Known driver bug (Error 802) as of Feb 2026."
    echo "Consider using A100 instances instead for stable training."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Clone repo if needed
if [ ! -d "nanollama" ]; then
    echo "Cloning nanollama..."
    git clone https://github.com/ariannamethod/nanollama.git
    cd nanollama
fi

# Create virtual environment and install dependencies
echo "Setting up Python environment..."
uv venv
source .venv/bin/activate
uv pip install -e ".[gpu]"

# Configure NCCL for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Create data directory
mkdir -p ~/.cache/nanollama/data

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  To start training:"
echo "    bash runs/speedrun.sh"
echo ""
echo "  To start smaller test:"
echo "    bash runs/runcpu.sh"
echo "========================================"
