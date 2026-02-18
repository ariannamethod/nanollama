#!/usr/bin/env python
"""
Extract personality vector (gamma) from trained nanollama models.

The core concept: after training a model WITH personality data mixed in,
we can extract the personality as a portable vector by comparing against
a model trained WITHOUT personality data.

Formula: γ = weights_with_personality - weights_without_personality

Usage:
    # Train base model (no personality)
    python scripts/base_train.py --depth 12 --personality_ratio 0.0 --out_dir run_base
    
    # Train personality model (same depth, same data + personality)
    python scripts/base_train.py --depth 12 --personality_ratio 0.2 --out_dir run_personality
    
    # Extract gamma
    python scripts/extract_gamma.py \\
        --personality_ckpt run_personality/ckpt.pt \\
        --base_ckpt run_base/ckpt.pt \\
        --output gamma_d12.npz
"""

import os
import argparse
from typing import Dict, Any

import torch
import numpy as np

from nanollama.common import print0


def parse_args():
    parser = argparse.ArgumentParser(description="Extract personality gamma vector")
    parser.add_argument("--personality_ckpt", type=str, required=True,
                        help="Checkpoint trained WITH personality data")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Checkpoint trained WITHOUT personality data (or early checkpoint)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for gamma NPZ file")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Minimum norm to include a layer (default: 1e-6)")
    parser.add_argument("--sparsity_threshold", type=float, default=1e-8,
                        help="Threshold for sparse storage (default: 1e-8)")
    return parser.parse_args()


def extract_gamma(
    checkpoint_with_personality: str,
    checkpoint_without_personality: str,
    threshold: float = 1e-6,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract gamma = personality weights - base weights.
    
    Args:
        checkpoint_with_personality: Path to checkpoint trained with personality
        checkpoint_without_personality: Path to checkpoint trained without personality
        threshold: Minimum norm to include a weight matrix
    
    Returns:
        Dictionary mapping layer names to {'diff', 'norm', 'sparsity'}
    """
    print0("Loading checkpoints...")
    
    # Load both checkpoints
    ckpt_personality = torch.load(checkpoint_with_personality, map_location='cpu', weights_only=False)
    ckpt_base = torch.load(checkpoint_without_personality, map_location='cpu', weights_only=False)
    
    # Get state dicts (handle both formats)
    # Handle _orig_mod. prefix from torch.compile
    if 'model_state_dict' in ckpt_personality:
        state_personality = ckpt_personality['model_state_dict']
    else:
        state_personality = ckpt_personality
    state_personality = {k.replace('_orig_mod.', ''): v for k, v in state_personality.items()}

    if 'model_state_dict' in ckpt_base:
        state_base = ckpt_base['model_state_dict']
    else:
        state_base = ckpt_base
    state_base = {k.replace('_orig_mod.', ''): v for k, v in state_base.items()}
    
    # Verify same architecture
    if set(state_personality.keys()) != set(state_base.keys()):
        print0("WARNING: Checkpoints have different architectures!")
        common_keys = set(state_personality.keys()) & set(state_base.keys())
        print0(f"Using {len(common_keys)} common keys")
    else:
        common_keys = set(state_personality.keys())
    
    print0(f"\nExtracting gamma from {len(common_keys)} weight matrices...")
    print0("-" * 60)
    
    gamma = {}
    total_norm = 0.0
    included_count = 0
    
    for key in sorted(common_keys):
        # Skip non-tensor entries
        if not isinstance(state_personality[key], torch.Tensor):
            continue
        
        # Compute difference
        w_personality = state_personality[key].float()
        w_base = state_base[key].float()
        
        # Check shape compatibility
        if w_personality.shape != w_base.shape:
            print0(f"  SKIP {key}: shape mismatch {w_personality.shape} vs {w_base.shape}")
            continue
        
        diff = w_personality - w_base
        norm = diff.norm().item()
        
        # Only store if meaningful change
        if norm > threshold:
            sparsity = (diff.abs() < 1e-8).float().mean().item()
            
            gamma[key] = {
                'diff': diff,
                'norm': norm,
                'sparsity': sparsity,
            }
            
            total_norm += norm ** 2
            included_count += 1
            
            print0(f"  {key}: norm={norm:.4f}, sparsity={sparsity*100:.1f}%")
    
    total_norm = total_norm ** 0.5
    print0("-" * 60)
    print0(f"Total gamma norm: {total_norm:.4f}")
    print0(f"Included layers: {included_count}/{len(common_keys)}")
    
    return gamma


def save_gamma_npz(
    gamma: Dict[str, Dict[str, Any]],
    output_path: str,
    sparsity_threshold: float = 1e-8,
):
    """
    Save gamma as sparse NPZ for efficient storage.
    
    Format:
        {key}.indices_0: row indices (int32)
        {key}.indices_1: col indices (int32) for 2D tensors
        {key}.values: non-zero values (float16)
        {key}.shape: original shape (int64)
    """
    print0(f"\nSaving gamma to {output_path}...")
    
    data = {}
    total_values = 0
    total_elements = 0
    
    for key, info in gamma.items():
        diff = info['diff']
        
        # Find non-zero (significant) values
        mask = diff.abs() > sparsity_threshold
        
        if mask.any():
            indices = mask.nonzero(as_tuple=True)
            values = diff[mask]
            
            # Store indices
            data[f"{key}.indices_0"] = indices[0].numpy().astype(np.int32)
            if len(indices) > 1:
                data[f"{key}.indices_1"] = indices[1].numpy().astype(np.int32)
            if len(indices) > 2:
                data[f"{key}.indices_2"] = indices[2].numpy().astype(np.int32)
            
            # Store values as float16 for compression
            data[f"{key}.values"] = values.numpy().astype(np.float16)
            
            # Store shape for reconstruction
            data[f"{key}.shape"] = np.array(diff.shape, dtype=np.int64)
            
            total_values += len(values)
            total_elements += diff.numel()
    
    # Add metadata (use descriptive names to avoid confusion with Python builtins)
    data["_metadata.layer_names"] = np.array(list(gamma.keys()), dtype=object)
    data["_metadata.total_norm"] = np.array(sum(info['norm'] ** 2 for info in gamma.values()) ** 0.5)
    
    # Save compressed
    np.savez_compressed(output_path, **data)
    
    # Report stats
    file_size = os.path.getsize(output_path)
    compression = total_values / total_elements if total_elements > 0 else 0
    
    print0(f"  Stored {total_values:,} / {total_elements:,} values ({compression*100:.1f}% density)")
    print0(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    print0("Done!")


def load_gamma_npz(input_path: str) -> Dict[str, torch.Tensor]:
    """
    Load gamma from NPZ file and reconstruct full tensors.
    
    Returns:
        Dictionary mapping layer names to gamma tensors
    """
    data = np.load(input_path, allow_pickle=True)
    
    # Get keys from metadata (check both old and new naming)
    if "_metadata.layer_names" in data:
        keys = data["_metadata.layer_names"]
    elif "_metadata.keys" in data:
        keys = data["_metadata.keys"]
    else:
        # Infer from stored data
        keys = set()
        for k in data.keys():
            if k.startswith("_metadata"):
                continue
            # Extract base key
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
        
        # Get indices
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


def main():
    args = parse_args()
    
    print0("=" * 60)
    print0("nanollama Gamma Extraction")
    print0("=" * 60)
    print0(f"Personality checkpoint: {args.personality_ckpt}")
    print0(f"Base checkpoint: {args.base_ckpt}")
    print0(f"Output: {args.output}")
    print0()
    
    # Extract gamma
    gamma = extract_gamma(
        args.personality_ckpt,
        args.base_ckpt,
        threshold=args.threshold,
    )
    
    if not gamma:
        print0("ERROR: No meaningful differences found between checkpoints!")
        return 1
    
    # Save gamma
    save_gamma_npz(gamma, args.output, sparsity_threshold=args.sparsity_threshold)
    
    return 0


if __name__ == "__main__":
    exit(main())
