"""
Checkpoint manager for nanollama.
Save and load model checkpoints.
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Tuple
from nanollama.common import get_base_dir, print0


def save_checkpoint(
    model,
    optimizer,
    step: int,
    config: dict,
    checkpoint_dir: str,
    name: str = "checkpoint",
):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        step: Current training step
        config: Model configuration dict
        checkpoint_dir: Directory to save to
        name: Checkpoint name prefix
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}_step{step}.pt")
    meta_path = os.path.join(checkpoint_dir, f"{name}_step{step}_meta.json")
    
    # Save model state dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'step': step,
        'config': config,
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata as JSON for easy inspection
    meta = {
        'step': step,
        'config': config,
        'checkpoint_path': checkpoint_path,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print0(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to
    
    Returns:
        Checkpoint dict with model_state_dict, optimizer_state_dict, step, config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pt') and 'step' in f:
            try:
                # Extract step number
                step = int(f.split('step')[1].split('.')[0])
                checkpoints.append((step, os.path.join(checkpoint_dir, f)))
            except (IndexError, ValueError):
                continue
    
    if not checkpoints:
        return None
    
    # Return checkpoint with highest step
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def load_model(
    model_tag: str,
    device: torch.device,
    phase: str = "train",
) -> Tuple[Any, Any, Dict]:
    """
    Load a model by tag name.
    
    Args:
        model_tag: Model identifier (e.g., "base", "chat", or a path)
        device: Device to load to
        phase: "train" or "eval" (affects torch.compile, etc.)
    
    Returns:
        (model, tokenizer, metadata)
    """
    from nanollama.llama import Llama, LlamaConfig
    from nanollama.tokenizer import get_tokenizer
    
    base_dir = get_base_dir()
    
    # Determine checkpoint path
    if os.path.exists(model_tag):
        checkpoint_path = model_tag
    else:
        checkpoint_dir = os.path.join(base_dir, "checkpoints", model_tag)
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found for model tag: {model_tag}")
    
    # Load checkpoint
    print0(f"Loading model from {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Create model from config
    config_dict = checkpoint.get('config', {})
    config = LlamaConfig(**config_dict)
    
    model = Llama(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if phase == "eval":
        model.eval()
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    
    # Metadata
    meta = {
        'step': checkpoint.get('step', 0),
        'config': config_dict,
        'checkpoint_path': checkpoint_path,
    }
    
    return model, tokenizer, meta


def save_for_inference(
    model,
    config: dict,
    output_dir: str,
    name: str = "model",
):
    """
    Save model in a format optimized for inference.
    
    Args:
        model: The model to save
        config: Model configuration
        output_dir: Output directory
        name: Model name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, f"{name}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save config
    config_path = os.path.join(output_dir, f"{name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print0(f"Saved inference model to {output_dir}")
