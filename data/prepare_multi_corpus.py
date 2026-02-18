"""
Multi-corpus data preparation following SmolLM2 recipe.

4 components, configurable ratios:
  - FineWeb-Edu (60%): High-quality educational web text
  - DCLM-Baseline (30%): Curated web corpus
  - The Stack v2 (10%): Code (deduped, permissive licenses)
  - MegaMath (bonus): Math reasoning data

Each component is streamed, tokenized, and saved as shards.
The dataloader mixes them at the specified ratios during training.

Usage:
    # Prepare all components (small run)
    python -m data.prepare_multi_corpus --total-tokens 100M

    # Only FineWeb-Edu + DCLM (no code/math)
    python -m data.prepare_multi_corpus --total-tokens 500M --components fineweb,dclm

    # Full run for serious training
    python -m data.prepare_multi_corpus --total-tokens 3B

    # Custom ratios
    python -m data.prepare_multi_corpus --total-tokens 1B --ratios 0.5,0.3,0.15,0.05

Requires: pip install datasets sentencepiece tqdm
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from nanollama.common import get_base_dir, print0


# Dataset configs: (HuggingFace ID, subset, text field, default ratio)
CORPUS_CONFIGS = {
    "fineweb": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "default_ratio": 0.60,
        "description": "FineWeb-Edu: educational web text (5.4T tokens total)",
    },
    "dclm": {
        "hf_id": "mlfoundations/dclm-baseline-1.0",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "default_ratio": 0.30,
        "description": "DCLM-Baseline: curated web corpus (4T tokens total)",
    },
    "stack": {
        "hf_id": "bigcode/the-stack-v2-dedup",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "default_ratio": 0.10,
        "description": "The Stack v2: deduped code (67.5TB total)",
    },
    "megamath": {
        "hf_id": "MHHMM/MegaMath",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "default_ratio": 0.0,  # Bonus, not default
        "description": "MegaMath: mathematical reasoning (371B tokens total)",
    },
}


def parse_token_count(s: str) -> int:
    """Parse human-readable token count: '100M', '1B', '500K'."""
    s = s.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def load_tokenizer(tokenizer_dir: str):
    """Load existing SentencePiece tokenizer."""
    import sentencepiece as spm

    model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {model_path}. "
            f"Run prepare_fineweb.py first to train the tokenizer."
        )
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp


def prepare_component(
    name: str,
    config: dict,
    sp,
    target_tokens: int,
    output_dir: str,
    shard_size: int = 10_000_000,
):
    """
    Download and tokenize one corpus component.

    Args:
        name: Component name (fineweb, dclm, stack, megamath)
        config: Dataset config dict
        sp: SentencePiece processor
        target_tokens: How many tokens to collect
        output_dir: Where to save shards
        shard_size: Tokens per shard
    """
    from datasets import load_dataset

    comp_dir = os.path.join(output_dir, name)
    os.makedirs(comp_dir, exist_ok=True)

    # Check if already prepared
    existing_shards = [f for f in os.listdir(comp_dir) if f.endswith('.bin') and f.startswith('train_')]
    if existing_shards:
        existing_tokens = sum(
            os.path.getsize(os.path.join(comp_dir, f)) // 2  # uint16 = 2 bytes
            for f in existing_shards
        )
        if existing_tokens >= target_tokens * 0.95:
            print0(f"  {name}: Already prepared ({existing_tokens:,} tokens in {len(existing_shards)} shards)")
            return comp_dir

    print0(f"\n  Loading {config['hf_id']}...")
    load_kwargs = {
        "path": config["hf_id"],
        "split": config["split"],
        "streaming": True,
    }
    if config["subset"]:
        load_kwargs["name"] = config["subset"]

    try:
        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        print0(f"  WARNING: Could not load {name}: {e}")
        print0(f"  Skipping {name}.")
        return None

    text_field = config["text_field"]
    shard_idx = 0
    token_buffer = []
    total_tokens = 0
    pbar = tqdm(total=target_tokens, desc=f"  {name}", unit="tok", unit_scale=True)

    for example in dataset:
        if total_tokens >= target_tokens:
            break

        text = example.get(text_field, "")
        if not text or len(text) < 50:  # Skip very short docs
            continue

        tokens = sp.encode(text)
        token_buffer.extend(tokens)
        pbar.update(len(tokens))

        while len(token_buffer) >= shard_size:
            shard_tokens = np.array(token_buffer[:shard_size], dtype=np.uint16)
            shard_path = os.path.join(comp_dir, f"train_{shard_idx:04d}.bin")
            shard_tokens.tofile(shard_path)
            token_buffer = token_buffer[shard_size:]
            total_tokens += shard_size
            shard_idx += 1

    pbar.close()

    # Save remaining
    if len(token_buffer) > 1000:
        shard_tokens = np.array(token_buffer, dtype=np.uint16)
        shard_path = os.path.join(comp_dir, f"train_{shard_idx:04d}.bin")
        shard_tokens.tofile(shard_path)
        total_tokens += len(token_buffer)
        shard_idx += 1

    print0(f"  {name}: {total_tokens:,} tokens in {shard_idx} shards -> {comp_dir}")
    return comp_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-corpus training data (SmolLM2 recipe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--total-tokens", type=str, default="100M",
                        help="Total tokens to prepare (e.g., 100M, 1B, 3B)")
    parser.add_argument("--components", type=str, default="fineweb,dclm,stack",
                        help="Comma-separated component names")
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated ratios (must match components count)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output base directory")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Tokenizer directory")
    parser.add_argument("--shard-size", type=int, default=10_000_000,
                        help="Tokens per shard")
    args = parser.parse_args()

    base_dir = get_base_dir()
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "data", "multi_corpus")
    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(base_dir, "tokenizer")

    total_tokens = parse_token_count(args.total_tokens)
    components = [c.strip() for c in args.components.split(",")]

    # Validate components
    for c in components:
        if c not in CORPUS_CONFIGS:
            print0(f"ERROR: Unknown component '{c}'. Available: {list(CORPUS_CONFIGS.keys())}")
            return 1

    # Parse or compute ratios
    if args.ratios:
        ratios = [float(r) for r in args.ratios.split(",")]
        if len(ratios) != len(components):
            print0(f"ERROR: {len(ratios)} ratios for {len(components)} components")
            return 1
    else:
        ratios = [CORPUS_CONFIGS[c]["default_ratio"] for c in components]
        # Normalize
        total_ratio = sum(ratios)
        if total_ratio > 0:
            ratios = [r / total_ratio for r in ratios]

    print0("=" * 60)
    print0("nanollama Multi-Corpus Data Preparation")
    print0("=" * 60)
    print0(f"Total tokens: {total_tokens:,}")
    print0(f"Output: {args.output_dir}")
    print0(f"Tokenizer: {args.tokenizer_dir}")
    print0()

    for c, r in zip(components, ratios):
        tokens_for = int(total_tokens * r)
        print0(f"  {c}: {r*100:.0f}% = {tokens_for:,} tokens")
        print0(f"    {CORPUS_CONFIGS[c]['description']}")
    print0()

    # Load tokenizer
    sp = load_tokenizer(args.tokenizer_dir)
    print0(f"Tokenizer: {sp.get_piece_size()} pieces")

    # Prepare each component
    os.makedirs(args.output_dir, exist_ok=True)
    prepared = {}

    for component, ratio in zip(components, ratios):
        target = int(total_tokens * ratio)
        if target < 1000:
            print0(f"  Skipping {component} (target too small: {target})")
            continue

        config = CORPUS_CONFIGS[component]
        result = prepare_component(
            name=component,
            config=config,
            sp=sp,
            target_tokens=target,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
        )
        if result:
            prepared[component] = result

    # Create merged symlinks or manifest
    import json
    manifest = {
        "components": {},
        "total_tokens": total_tokens,
        "vocab_size": sp.get_piece_size(),
        "tokenizer_dir": args.tokenizer_dir,
    }
    for c, path in prepared.items():
        shards = sorted([f for f in os.listdir(path) if f.endswith('.bin')])
        tokens = sum(os.path.getsize(os.path.join(path, f)) // 2 for f in shards)
        manifest["components"][c] = {
            "path": path,
            "num_shards": len(shards),
            "tokens": tokens,
        }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Create a flat data directory with all shards (for simple dataloader)
    merged_dir = os.path.join(args.output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    shard_count = 0
    for c, path in prepared.items():
        for f in sorted(os.listdir(path)):
            if f.endswith('.bin'):
                src = os.path.join(path, f)
                dst = os.path.join(merged_dir, f"{c}_{f}")
                if not os.path.exists(dst):
                    os.symlink(src, dst)
                shard_count += 1

    print0(f"\n{'='*60}")
    print0(f"Done! {shard_count} total shards")
    print0(f"Manifest: {manifest_path}")
    print0(f"Merged dir: {merged_dir}")
    print0(f"\nTo train with merged data:")
    print0(f"  python -m scripts.base_train --depth=12 --vocab-size={sp.get_piece_size()} --data-dir={merged_dir}")
    print0(f"\nTo add personality:")
    print0(f"  --personality-dir=<personality_dir> --personality-ratio=0.2")

    return 0


if __name__ == "__main__":
    exit(main())
