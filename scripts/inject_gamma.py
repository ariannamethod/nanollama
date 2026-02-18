#!/usr/bin/env python
"""
Inject personality gamma vector into a nanollama model.

The injection formula: θ = ε + α×γ
Where:
    θ = output model weights
    ε = base model weights  
    γ = personality gamma vector
    α = injection strength (typically 0.5-1.0)

For same-depth injection, it's straightforward addition.
For cross-depth (e.g., d12 gamma into d20 model):
    - Match layers by relative position (layer i of N maps to layer j of M)
    - Only inject into layers that exist in both
    - Skip layers with dimension mismatch

Usage:
    # Same depth injection
    python scripts/inject_gamma.py \\
        --base_ckpt run_base/ckpt.pt \\
        --gamma gamma_d12.npz \\
        --alpha 0.5 \\
        --output model_with_personality.pt
    
    # Cross-depth injection (experimental)
    python scripts/inject_gamma.py \\
        --base_ckpt run_base_d20/ckpt.pt \\
        --gamma gamma_d12.npz \\
        --alpha 0.5 \\
        --cross_depth \\
        --output model_d20_with_personality.pt
"""

import os
import re
import argparse
from typing import Dict, Optional, Tuple

import torch
import numpy as np

from nanollama.common import print0


def parse_args():
    parser = argparse.ArgumentParser(description="Inject personality gamma into model")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Base checkpoint to inject into")
    parser.add_argument("--gamma", type=str, required=True,
                        help="Gamma NPZ file from extract_gamma.py")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Injection strength (default: 1.0)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output checkpoint path")
    parser.add_argument("--cross_depth", action="store_true",
                        help="Enable cross-depth injection (experimental)")
    return parser.parse_args()


def load_gamma_npz(input_path: str) -> Dict[str, torch.Tensor]:
    """Load gamma from NPZ file and reconstruct full tensors."""
    data = np.load(input_path, allow_pickle=True)
    
    # Get keys from metadata (check both old and new naming)
    if "_metadata.layer_names" in data:
        keys = data["_metadata.layer_names"]
    elif "_metadata.keys" in data:
        keys = data["_metadata.keys"]
    else:
        keys = set()
        for k in data.keys():
            if k.startswith("_metadata"):
                continue
            parts = k.rsplit(".", 1)
            if len(parts) == 2 and parts[1] in ("indices_0", "indices_1", "indices_2", "values", "shape"):
                keys.add(parts[0])
        keys = sorted(keys)
    
    gamma = {}
    for key in keys:
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        
        shape_key = f"{key}.shape"
        values_key = f"{key}.values"
        
        if shape_key not in data or values_key not in data:
            continue
        
        shape = tuple(data[shape_key])
        values = torch.from_numpy(data[values_key].astype(np.float32))
        
        # Reconstruct tensor
        tensor = torch.zeros(shape, dtype=torch.float32)
        
        idx0 = torch.from_numpy(data[f"{key}.indices_0"].astype(np.int64))
        
        if f"{key}.indices_1" in data:
            idx1 = torch.from_numpy(data[f"{key}.indices_1"].astype(np.int64))
            if f"{key}.indices_2" in data:
                idx2 = torch.from_numpy(data[f"{key}.indices_2"].astype(np.int64))
                tensor[idx0, idx1, idx2] = values
            else:
                tensor[idx0, idx1] = values
        else:
            tensor[idx0] = values
        
        gamma[key] = tensor
    
    return gamma


def get_layer_idx(key: str) -> Optional[int]:
    """Extract layer index from key like 'layers.5.attn.c_q.weight'"""
    match = re.match(r'layers\.(\d+)\.', key)
    if match:
        return int(match.group(1))
    return None


def remap_layer_key(key: str, source_idx: int, target_idx: int) -> str:
    """Remap layer index in key."""
    return key.replace(f'layers.{source_idx}.', f'layers.{target_idx}.')


def compute_layer_mapping(gamma_n_layers: int, model_n_layers: int) -> Dict[int, int]:
    """
    Compute layer mapping for cross-depth injection.
    Maps source layer i to target layer j where j/M ≈ i/N
    
    Args:
        gamma_n_layers: Number of layers in gamma source
        model_n_layers: Number of layers in target model
    
    Returns:
        Dict mapping source layer idx -> target layer idx
    """
    mapping = {}
    for i in range(gamma_n_layers):
        # Proportional mapping
        ratio = i / (gamma_n_layers - 1) if gamma_n_layers > 1 else 0
        j = int(round(ratio * (model_n_layers - 1)))
        mapping[i] = j
    return mapping


def inject_gamma(
    base_ckpt_path: str,
    gamma: Dict[str, torch.Tensor],
    alpha: float = 1.0,
    cross_depth: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Inject gamma into base checkpoint.
    
    Args:
        base_ckpt_path: Path to base checkpoint
        gamma: Gamma tensors from load_gamma_npz
        alpha: Injection strength
        cross_depth: Enable cross-depth mapping
    
    Returns:
        (modified_checkpoint, stats)
    """
    print0("Loading base checkpoint...")
    ckpt = torch.load(base_ckpt_path, map_location='cpu', weights_only=False)
    
    # Get state dict — handle _orig_mod. prefix from torch.compile
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
        ckpt = {'model_state_dict': state}
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    
    # Count layers in both
    gamma_layers = set()
    model_layers = set()
    
    for key in gamma.keys():
        idx = get_layer_idx(key)
        if idx is not None:
            gamma_layers.add(idx)
    
    for key in state.keys():
        idx = get_layer_idx(key)
        if idx is not None:
            model_layers.add(idx)
    
    gamma_n_layers = max(gamma_layers) + 1 if gamma_layers else 0
    model_n_layers = max(model_layers) + 1 if model_layers else 0
    
    print0(f"Gamma layers: {gamma_n_layers}")
    print0(f"Model layers: {model_n_layers}")
    
    # Compute layer mapping if cross-depth
    layer_mapping = None
    if cross_depth and gamma_n_layers != model_n_layers:
        layer_mapping = compute_layer_mapping(gamma_n_layers, model_n_layers)
        print0(f"Cross-depth mapping: {layer_mapping}")
    
    print0(f"\nInjecting gamma with alpha={alpha}...")
    print0("-" * 60)
    
    stats = {
        'injected': 0,
        'skipped_shape': 0,
        'skipped_missing': 0,
        'total_gamma_norm': 0.0,
    }
    
    for gamma_key, gamma_tensor in gamma.items():
        target_key = gamma_key
        
        # Remap layer if cross-depth
        if layer_mapping is not None:
            source_idx = get_layer_idx(gamma_key)
            if source_idx is not None and source_idx in layer_mapping:
                target_idx = layer_mapping[source_idx]
                target_key = remap_layer_key(gamma_key, source_idx, target_idx)
        
        # Check if key exists in model
        if target_key not in state:
            print0(f"  SKIP {gamma_key}: not in target model")
            stats['skipped_missing'] += 1
            continue
        
        # Check shape compatibility
        if state[target_key].shape != gamma_tensor.shape:
            print0(f"  SKIP {gamma_key}: shape mismatch "
                   f"{gamma_tensor.shape} vs {state[target_key].shape}")
            stats['skipped_shape'] += 1
            continue
        
        # Inject: θ = ε + α×γ (convert gamma to match state dtype to avoid unnecessary conversions)
        state[target_key] = state[target_key] + alpha * gamma_tensor.to(state[target_key].dtype)
        
        gamma_norm = gamma_tensor.norm().item()
        stats['total_gamma_norm'] += gamma_norm ** 2
        stats['injected'] += 1
        
        print0(f"  ✓ {gamma_key} (norm={gamma_norm:.4f})")
    
    stats['total_gamma_norm'] = stats['total_gamma_norm'] ** 0.5
    
    print0("-" * 60)
    print0(f"Injected: {stats['injected']}")
    print0(f"Skipped (shape mismatch): {stats['skipped_shape']}")
    print0(f"Skipped (missing in model): {stats['skipped_missing']}")
    print0(f"Total gamma norm: {stats['total_gamma_norm']:.4f}")
    print0(f"Effective injection: alpha * norm = {alpha * stats['total_gamma_norm']:.4f}")
    
    # Update checkpoint
    if 'model_state_dict' in ckpt:
        ckpt['model_state_dict'] = state
    else:
        ckpt = state
    
    return ckpt, stats


def main():
    args = parse_args()
    
    print0("=" * 60)
    print0("nanollama Gamma Injection")
    print0("=" * 60)
    print0(f"Base checkpoint: {args.base_ckpt}")
    print0(f"Gamma file: {args.gamma}")
    print0(f"Alpha: {args.alpha}")
    print0(f"Cross-depth: {args.cross_depth}")
    print0(f"Output: {args.output}")
    print0()
    
    # Load gamma
    print0("Loading gamma...")
    gamma = load_gamma_npz(args.gamma)
    print0(f"Loaded {len(gamma)} gamma tensors")
    print0()
    
    # Inject
    ckpt, stats = inject_gamma(
        args.base_ckpt,
        gamma,
        alpha=args.alpha,
        cross_depth=args.cross_depth,
    )
    
    if stats['injected'] == 0:
        print0("\nERROR: No layers were injected!")
        return 1
    
    # Save
    print0(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(ckpt, args.output)
    
    file_size = os.path.getsize(args.output)
    print0(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print0("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
