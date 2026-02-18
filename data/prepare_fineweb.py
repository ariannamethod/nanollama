"""
Download and prepare FineWeb-Edu dataset for pretraining.

FineWeb-Edu is a filtered version of FineWeb focused on educational content.
Downloads streaming data, trains a SentencePiece tokenizer, and saves tokenized shards.

Usage:
    # Quick proof of concept (200K samples, ~50M tokens)
    python -m data.prepare_fineweb --num-samples 200000

    # Full training run (10M samples, ~3B tokens)
    python -m data.prepare_fineweb --num-samples 10000000

    # Custom output and vocab
    python -m data.prepare_fineweb --num-samples 500000 --vocab-size 32000 --output-dir ./data/fineweb

Adapted from nanochat for nanollama.
"""

import os
import argparse
import tempfile
import numpy as np
from tqdm import tqdm

from nanollama.common import get_base_dir, print0


def train_tokenizer(text_iter, vocab_size: int, output_dir: str):
    """Train a SentencePiece BPE tokenizer from text iterator."""
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "tokenizer.model")

    if os.path.exists(model_path):
        print0(f"Tokenizer already exists at {model_path}, skipping training")
        sp = spm.SentencePieceProcessor(model_file=model_path)
        return sp

    # Write texts to temp file for SentencePiece training
    print0(f"Training SentencePiece tokenizer (vocab_size={vocab_size})...")
    tmp_file = os.path.join(output_dir, "_train_text.txt")
    n_written = 0
    with open(tmp_file, 'w', encoding='utf-8') as f:
        for text in text_iter:
            f.write(text.strip() + '\n')
            n_written += 1
    print0(f"Wrote {n_written:,} documents for tokenizer training")

    model_prefix = os.path.join(output_dir, "tokenizer")
    spm.SentencePieceTrainer.train(
        input=tmp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        num_threads=os.cpu_count() or 4,
        split_digits=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        normalization_rule_name="identity",
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        input_sentence_size=min(n_written, 5_000_000),
    )

    # Cleanup temp file
    os.unlink(tmp_file)

    sp = spm.SentencePieceProcessor(model_file=model_path)
    print0(f"Tokenizer trained: {sp.get_piece_size()} pieces -> {model_path}")
    return sp


def prepare_fineweb(
    num_samples: int = 200_000,
    output_dir: str = None,
    tokenizer_dir: str = None,
    vocab_size: int = 32000,
    shard_size: int = 10_000_000,  # tokens per shard
    tokenizer_train_samples: int = 50_000,  # samples for tokenizer training
):
    """
    Download FineWeb-Edu, train tokenizer, tokenize into shards.

    Args:
        num_samples: Total number of FineWeb-Edu documents to download
        output_dir: Where to save tokenized shards
        tokenizer_dir: Where to save/load tokenizer
        vocab_size: Vocabulary size for SentencePiece
        shard_size: Number of tokens per shard file
        tokenizer_train_samples: How many docs to use for tokenizer training
    """
    from datasets import load_dataset

    base_dir = get_base_dir()

    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "fineweb")
    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(base_dir, "tokenizer")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load dataset (streaming)
    print0(f"Loading FineWeb-Edu (streaming, {num_samples:,} samples)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Step 2: Train tokenizer on a subset
    print0(f"\n--- Step 1/3: Train tokenizer on {tokenizer_train_samples:,} samples ---")

    tokenizer_model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    if os.path.exists(tokenizer_model_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        print0(f"Loaded existing tokenizer: {sp.get_piece_size()} pieces")
        # We still need to consume the dataset from the beginning
        texts_for_tokenizer = []
    else:
        # Collect texts for tokenizer training
        print0("Collecting texts for tokenizer training...")
        texts_for_tokenizer = []
        dataset_iter = iter(dataset)
        for i in tqdm(range(tokenizer_train_samples), desc="Collecting"):
            try:
                example = next(dataset_iter)
                text = example.get("text", "")
                if text:
                    texts_for_tokenizer.append(text)
            except StopIteration:
                break

        sp = train_tokenizer(iter(texts_for_tokenizer), vocab_size, tokenizer_dir)

    actual_vocab_size = sp.get_piece_size()
    print0(f"Tokenizer vocab size: {actual_vocab_size}")

    # Step 3: Tokenize all data into shards
    print0(f"\n--- Step 2/3: Tokenize {num_samples:,} samples into shards ---")

    # Re-load dataset fresh for full tokenization
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    shard_idx = 0
    token_buffer = []
    total_tokens = 0
    pbar = tqdm(total=num_samples, desc="Tokenizing")

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        text = example.get("text", "")
        if not text:
            continue

        tokens = sp.encode(text)
        token_buffer.extend(tokens)
        pbar.update(1)

        # Flush shard when buffer is full
        while len(token_buffer) >= shard_size:
            shard_tokens = np.array(token_buffer[:shard_size], dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
            shard_tokens.tofile(shard_path)
            print0(f"  Shard {shard_idx}: {len(shard_tokens):,} tokens -> {shard_path}")

            token_buffer = token_buffer[shard_size:]
            total_tokens += shard_size
            shard_idx += 1

    pbar.close()

    # Save remaining tokens as last shard (if meaningful)
    if len(token_buffer) > 1000:
        shard_tokens = np.array(token_buffer, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
        shard_tokens.tofile(shard_path)
        total_tokens += len(token_buffer)
        shard_idx += 1

    print0(f"\nTotal: {total_tokens:,} tokens in {shard_idx} shards")

    # Step 4: Create a small val set from the end
    print0(f"\n--- Step 3/3: Create validation set ---")

    # Use 5% of last shard as validation
    val_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    val_tokens = []
    val_target = max(shard_size // 2, 500_000)  # At least 500K val tokens
    # Skip to samples after training set
    for i, example in enumerate(val_dataset):
        if i < num_samples:
            continue
        if i >= num_samples + 50_000:
            break

        text = example.get("text", "")
        if text:
            val_tokens.extend(sp.encode(text))
            if len(val_tokens) >= val_target:
                break

    if val_tokens:
        val_arr = np.array(val_tokens[:val_target], dtype=np.uint16)
        val_path = os.path.join(output_dir, "val_0000.bin")
        val_arr.tofile(val_path)
        print0(f"Validation: {len(val_arr):,} tokens -> {val_path}")

    # Save metadata
    import json
    meta = {
        "dataset": "fineweb-edu/sample-10BT",
        "vocab_size": actual_vocab_size,
        "total_tokens": total_tokens,
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "num_samples": num_samples,
        "tokenizer_dir": tokenizer_dir,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print0(f"\nDone! Data in {output_dir}")
    print0(f"Tokenizer in {tokenizer_dir}")
    print0(f"\nTo train:")
    print0(f"  python -m scripts.base_train --depth=12 --vocab-size={actual_vocab_size} --data-dir={output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    parser.add_argument("--num-samples", type=int, default=200_000,
                        help="Number of documents to download (200K ≈ 50M tokens)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for tokenized shards")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Tokenizer directory (trains new if not found)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size for tokenizer")
    parser.add_argument("--shard-size", type=int, default=10_000_000,
                        help="Tokens per shard")
    parser.add_argument("--tokenizer-train-samples", type=int, default=50_000,
                        help="Documents to use for tokenizer training")
    args = parser.parse_args()

    prepare_fineweb(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        shard_size=args.shard_size,
        tokenizer_train_samples=args.tokenizer_train_samples,
    )
