"""
Tokenizer training script for nanollama.

Train a SentencePiece tokenizer on your data.

Usage:
    python -m scripts.tok_train --input=data.txt --vocab-size=32000
"""

import argparse
import os

from nanollama.common import get_base_dir, print0
from nanollama.tokenizer import SentencePieceTokenizer, SPECIAL_TOKENS


def parse_args():
    parser = argparse.ArgumentParser(description="Train nanollama tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input text file or directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def iterate_texts(input_path: str):
    """Iterate over texts from file or directory."""
    if os.path.isfile(input_path):
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(input_path, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line


def main():
    args = parse_args()
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(get_base_dir(), "tokenizer")
    
    print0(f"Training tokenizer with vocab_size={args.vocab_size}")
    print0(f"Input: {args.input}")
    print0(f"Output: {args.output_dir}")
    
    # Train tokenizer
    tokenizer = SentencePieceTokenizer.train_from_iterator(
        iterate_texts(args.input),
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )
    
    # Save
    tokenizer.save(args.output_dir)
    
    # Test
    print0("\nTesting tokenizer:")
    test_text = "Hello, world! This is a test of the nanollama tokenizer."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print0(f"  Input: {test_text}")
    print0(f"  Tokens: {tokens[:20]}...")
    print0(f"  Decoded: {decoded}")
    print0(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    print0("\nDone!")


if __name__ == "__main__":
    main()
