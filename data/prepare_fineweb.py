"""
Download and prepare FineWeb-Edu dataset for pretraining.

FineWeb-Edu is a filtered version of FineWeb focused on educational content.
This script downloads, tokenizes, and saves the data in binary format.

Usage:
    python -m data.prepare_fineweb --num-samples 1000000

Adapted from nanochat for nanollama.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from nanollama.common import get_base_dir, print0


def prepare_fineweb(
    num_samples: int = 1_000_000,
    output_dir: str = None,
    tokenizer_path: str = None,
):
    """Download and tokenize FineWeb-Edu dataset."""
    from datasets import load_dataset
    from nanollama.tokenizer import Tokenizer
    
    if output_dir is None:
        output_dir = os.path.join(get_base_dir(), "data", "fineweb")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = Tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Tokenizer vocab size: {vocab_size}")
    
    # Load FineWeb-Edu dataset
    print0(f"Loading FineWeb-Edu (streaming, {num_samples:,} samples)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample
        split="train",
        streaming=True
    )
    
    # Tokenize and save
    all_tokens = []
    pbar = tqdm(total=num_samples, desc="Tokenizing")
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        text = example.get("text", "")
        if not text:
            continue
        
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        pbar.update(1)
    
    pbar.close()
    
    # Convert to numpy array
    total_tokens = len(all_tokens)
    print0(f"Total tokens: {total_tokens:,}")
    
    # Split 90/10 for train/val
    split_idx = int(0.9 * total_tokens)
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(all_tokens[split_idx:], dtype=np.uint16)
    
    # Save
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    
    print0(f"Train tokens: {len(train_tokens):,} -> {train_path}")
    print0(f"Val tokens: {len(val_tokens):,} -> {val_path}")
    
    # Save metadata
    meta_path = os.path.join(output_dir, "meta.json")
    import json
    with open(meta_path, 'w') as f:
        json.dump({
            "vocab_size": vocab_size,
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
            "total_tokens": total_tokens,
        }, f, indent=2)
    
    print0(f"Metadata saved to {meta_path}")
    print0("Done!")
    
    return output_dir


def prepare_fineweb_shards(
    num_shards: int = 100,
    tokens_per_shard: int = 10_000_000,
    output_dir: str = None,
):
    """Prepare FineWeb in shards for distributed training."""
    from datasets import load_dataset
    from nanollama.tokenizer import Tokenizer
    
    if output_dir is None:
        output_dir = os.path.join(get_base_dir(), "data", "fineweb_shards")
    
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = Tokenizer()
    
    print0(f"Loading FineWeb-Edu (streaming)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    shard_idx = 0
    tokens_buffer = []
    
    for example in tqdm(dataset, desc="Processing"):
        if shard_idx >= num_shards:
            break
        
        text = example.get("text", "")
        if not text:
            continue
        
        tokens = tokenizer.encode(text)
        tokens_buffer.extend(tokens)
        
        # Save shard when buffer is full
        if len(tokens_buffer) >= tokens_per_shard:
            shard_tokens = np.array(tokens_buffer[:tokens_per_shard], dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
            shard_tokens.tofile(shard_path)
            print0(f"Saved shard {shard_idx}: {len(shard_tokens):,} tokens")
            
            tokens_buffer = tokens_buffer[tokens_per_shard:]
            shard_idx += 1
    
    print0(f"Created {shard_idx} shards in {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    parser.add_argument("--num-samples", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--shards", action="store_true", help="Create sharded format")
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    args = parser.parse_args()
    
    if args.shards:
        prepare_fineweb_shards(args.num_shards, args.tokens_per_shard, args.output_dir)
    else:
        prepare_fineweb(args.num_samples, args.output_dir)
