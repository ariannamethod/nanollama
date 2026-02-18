"""
Prepare personality data for nanollama pretraining injection.

Tokenizes personality conversations as plain text (no chat template needed).
For pretraining, personality is injected as a portion of each batch —
the model learns the content/style, not the chat format.

Supports multiple input formats:
  - instruction/response JSONL (Yent dataset)
  - messages JSONL (chat format)
  - plain text files

Usage:
    # From Yent JSONL
    python -m data.prepare_personality --input yent_dataset.jsonl

    # Specify tokenizer directory
    python -m data.prepare_personality --input yent_dataset.jsonl --tokenizer-dir ~/.cache/nanollama/tokenizer

    # From plain text
    python -m data.prepare_personality --input personality.txt --format text
"""

import os
import json
import argparse
import numpy as np

from nanollama.common import get_base_dir, print0


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare personality data")
    parser.add_argument("--input", type=str, required=True, help="Input file (JSONL or text)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Tokenizer directory (must exist, run prepare_fineweb first)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "jsonl", "text"],
                        help="Input format (default: auto-detect)")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator between instruction and response")
    return parser.parse_args()


def load_texts_from_jsonl(input_path: str, separator: str = "\n\n"):
    """Extract plain text from JSONL conversations."""
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            parts = []

            # Format 0: bare messages array [{"role": ..., "content": ...}, ...]
            if isinstance(item, list):
                for msg in item:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if content:
                            parts.append(content)

            # Format 1: instruction/response (Yent dataset)
            elif "instruction" in item and "response" in item:
                parts.append(item["instruction"])
                parts.append(item["response"])

            # Format 2: user/assistant
            elif "user" in item and "assistant" in item:
                parts.append(item["user"])
                parts.append(item["assistant"])

            # Format 3: messages list
            elif "messages" in item:
                for msg in item["messages"]:
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)

            # Format 4: just "text" field
            elif "text" in item:
                parts.append(item["text"])

            else:
                continue

            if parts:
                texts.append(separator.join(parts))

    return texts


def load_texts_from_file(input_path: str):
    """Load plain text, split on double newlines."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split into documents on double newlines
    docs = [d.strip() for d in content.split("\n\n\n") if d.strip()]
    return docs


def main():
    args = parse_args()

    base_dir = get_base_dir()
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "data", "personality")
    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(base_dir, "tokenizer")

    os.makedirs(args.output_dir, exist_ok=True)

    print0("=" * 60)
    print0("nanollama Personality Data Preparation")
    print0("=" * 60)
    print0(f"Input: {args.input}")
    print0(f"Output: {args.output_dir}")
    print0(f"Tokenizer: {args.tokenizer_dir}")
    print0()

    # Load tokenizer
    import sentencepiece as spm
    tokenizer_path = os.path.join(args.tokenizer_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        print0(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print0("Run prepare_fineweb.py first to train the tokenizer.")
        return 1

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print0(f"Loaded tokenizer: {sp.get_piece_size()} pieces")

    # Detect format
    fmt = args.format
    if fmt == "auto":
        if args.input.endswith(".jsonl") or args.input.endswith(".json"):
            fmt = "jsonl"
        else:
            fmt = "text"
    print0(f"Input format: {fmt}")

    # Load texts
    if fmt == "jsonl":
        texts = load_texts_from_jsonl(args.input, args.separator)
    else:
        texts = load_texts_from_file(args.input)

    if not texts:
        print0("ERROR: No texts found in input file!")
        return 1

    print0(f"Loaded {len(texts):,} documents")

    # Tokenize
    print0("Tokenizing...")
    all_tokens = []
    for text in texts:
        tokens = sp.encode(text)
        all_tokens.extend(tokens)

    print0(f"Total tokens: {len(all_tokens):,}")
    print0(f"Avg tokens/doc: {len(all_tokens) / len(texts):.0f}")

    # Save as binary shard
    output_path = os.path.join(args.output_dir, "personality_000.bin")
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    tokens_array.tofile(output_path)

    file_size = os.path.getsize(output_path)
    print0(f"\nSaved: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
    print0(f"  Documents: {len(texts):,}")
    print0(f"  Tokens: {len(all_tokens):,}")
    print0(f"  Vocab size: {sp.get_piece_size()}")

    print0(f"\nTo use during training:")
    print0(f"  --personality-dir={args.output_dir} --personality-ratio=0.2")

    return 0


if __name__ == "__main__":
    exit(main())
