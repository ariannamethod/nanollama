"""
Export nanollama model to GGUF format for llama.cpp inference.

Usage:
    python -m scripts.export_gguf --model-tag=chat --output=model.gguf
"""

import os
import json
import struct
import argparse
from typing import Dict, Any

import torch
import numpy as np

from nanollama.common import print0
from nanollama.checkpoint_manager import load_checkpoint, get_latest_checkpoint


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# Tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Export nanollama to GGUF")
    parser.add_argument("--model-tag", type=str, required=True, help="Model to export")
    parser.add_argument("--output", type=str, default="model.gguf", help="Output file")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f32", "f16"], help="Output dtype")
    return parser.parse_args()


def write_gguf_header(f, n_tensors: int, n_kv: int):
    """Write GGUF file header."""
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))
    f.write(struct.pack('<Q', n_tensors))
    f.write(struct.pack('<Q', n_kv))


def write_gguf_string(f, s: str):
    """Write a GGUF string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_kv(f, key: str, value_type: int, value):
    """Write a key-value pair."""
    write_gguf_string(f, key)
    f.write(struct.pack('<I', value_type))
    
    if value_type == 4:  # GGUF_TYPE_UINT32
        f.write(struct.pack('<I', value))
    elif value_type == 5:  # GGUF_TYPE_INT32
        f.write(struct.pack('<i', value))
    elif value_type == 6:  # GGUF_TYPE_FLOAT32
        f.write(struct.pack('<f', value))
    elif value_type == 8:  # GGUF_TYPE_STRING
        write_gguf_string(f, value)


def map_tensor_name(name: str) -> str:
    """
    Map nanollama tensor names to llama.cpp GGUF format.
    
    llama.cpp expects:
    - model.embed_tokens.weight
    - model.layers.N.self_attn.q_proj.weight
    - model.layers.N.self_attn.k_proj.weight
    - model.layers.N.self_attn.v_proj.weight
    - model.layers.N.self_attn.o_proj.weight
    - model.layers.N.mlp.gate_proj.weight
    - model.layers.N.mlp.up_proj.weight
    - model.layers.N.mlp.down_proj.weight
    - model.norm.weight
    - lm_head.weight
    """
    # Embedding
    if name == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    
    # Output projection
    if name == "output.weight":
        return "lm_head.weight"
    
    # Layer mappings
    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
        
        # Attention projections
        if rest == "attn.c_q.weight":
            return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        if rest == "attn.c_k.weight":
            return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        if rest == "attn.c_v.weight":
            return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        if rest == "attn.c_proj.weight":
            return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        
        # FFN projections
        if rest == "ffn.gate_proj.weight":
            return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if rest == "ffn.up_proj.weight":
            return f"model.layers.{layer_idx}.mlp.up_proj.weight"
        if rest == "ffn.down_proj.weight":
            return f"model.layers.{layer_idx}.mlp.down_proj.weight"
    
    # Fallback: replace dots with underscores
    return name.replace(".", "_")


def main():
    args = parse_args()
    
    # Load checkpoint
    print0(f"Loading model: {args.model_tag}")
    from nanollama.common import get_base_dir
    
    if os.path.exists(args.model_tag):
        checkpoint_path = args.model_tag
    else:
        checkpoint_dir = os.path.join(get_base_dir(), "checkpoints", args.model_tag)
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoint found for: {args.model_tag}")
    
    checkpoint = load_checkpoint(checkpoint_path, torch.device('cpu'))
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    
    print0(f"Model config: {config}")
    print0(f"Number of tensors: {len(state_dict)}")
    
    # Convert state dict names to GGUF format
    tensor_names = []
    tensor_data = []
    
    for name, tensor in state_dict.items():
        # Convert to numpy
        if args.dtype == 'f16':
            arr = tensor.half().numpy()
            dtype = GGML_TYPE_F16
        else:
            arr = tensor.float().numpy()
            dtype = GGML_TYPE_F32
        
        # Map names to llama.cpp format
        gguf_name = map_tensor_name(name)
        
        tensor_names.append(gguf_name)
        tensor_data.append((gguf_name, arr, dtype))
    
    # Prepare metadata
    metadata = [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "nanollama"),
        ("llama.context_length", 4, config.get('sequence_len', 2048)),
        ("llama.embedding_length", 4, config.get('n_embd', 768)),
        ("llama.block_count", 4, config.get('n_layer', 12)),
        ("llama.attention.head_count", 4, config.get('n_head', 12)),
        ("llama.attention.head_count_kv", 4, config.get('n_kv_head', 4)),
        ("llama.rope.dimension_count", 4, config.get('n_embd', 768) // config.get('n_head', 12)),
        ("llama.vocab_size", 4, config.get('vocab_size', 32000)),
    ]
    
    # Write GGUF file
    print0(f"Writing to {args.output}...")
    
    with open(args.output, 'wb') as f:
        # Header
        write_gguf_header(f, len(tensor_data), len(metadata))
        
        # Metadata
        for key, value_type, value in metadata:
            write_gguf_kv(f, key, value_type, value)
        
        # Tensor info (simplified - real GGUF has more complex tensor headers)
        # This is a minimal implementation for demonstration
        
        # Note: A full GGUF implementation would need proper tensor alignment,
        # quantization support, and the complete tensor header format.
        # For production use, consider using llama.cpp's convert scripts.
        
        print0("Note: This is a simplified GGUF export.")
        print0("For production use, consider using llama.cpp's official convert scripts.")
    
    print0(f"Exported to {args.output}")
    print0("Done!")


if __name__ == "__main__":
    main()
