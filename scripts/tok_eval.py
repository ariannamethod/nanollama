"""
Tokenizer evaluation script for nanollama.

Evaluates tokenizer quality by computing:
- Compression ratio (tokens per character)
- Out-of-vocabulary rate
- Encoding/decoding speed

Usage:
    python -m scripts.tok_eval
"""

import os
import argparse
import time

from nanollama.tokenizer import get_tokenizer
from nanollama.common import print0


def evaluate_tokenizer(tokenizer, text: str, name: str = "test"):
    """Evaluate tokenizer on sample text."""
    print0(f"\n--- Evaluating on {name} ({len(text):,} chars) ---")
    
    # Encoding
    t0 = time.time()
    tokens = tokenizer.encode(text)
    encode_time = time.time() - t0
    
    # Decoding
    t0 = time.time()
    decoded = tokenizer.decode(tokens)
    decode_time = time.time() - t0
    
    # Stats
    num_chars = len(text)
    num_tokens = len(tokens)
    compression = num_chars / num_tokens if num_tokens > 0 else 0
    
    print0(f"  Characters: {num_chars:,}")
    print0(f"  Tokens: {num_tokens:,}")
    print0(f"  Compression ratio: {compression:.2f} chars/token")
    print0(f"  Encode time: {encode_time*1000:.2f}ms")
    print0(f"  Decode time: {decode_time*1000:.2f}ms")
    print0(f"  Encode speed: {num_chars/encode_time/1000:.1f}K chars/s")
    
    # Check roundtrip
    if decoded == text:
        print0("  Roundtrip: ✓ Perfect")
    else:
        diff_chars = sum(1 for a, b in zip(decoded, text) if a != b)
        print0(f"  Roundtrip: ✗ {diff_chars} differences")
    
    return {
        'name': name,
        'chars': num_chars,
        'tokens': num_tokens,
        'compression': compression,
        'encode_time': encode_time,
        'decode_time': decode_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer")
    parser.add_argument("--text-file", type=str, default=None, help="Text file to evaluate")
    args = parser.parse_args()
    
    print0("nanollama Tokenizer Evaluation")
    print0("=" * 50)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    print0(f"Tokenizer: {tokenizer.__class__.__name__}")
    print0(f"Vocab size: {tokenizer.get_vocab_size():,}")
    
    # Sample texts
    samples = [
        ("English", "The quick brown fox jumps over the lazy dog. " * 100),
        ("Code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 50),
        ("Numbers", "1234567890 " * 200),
        ("Mixed", "Hello world! Привет мир! 你好世界! 🎉🚀" * 50),
    ]
    
    results = []
    for name, text in samples:
        result = evaluate_tokenizer(tokenizer, text, name)
        results.append(result)
    
    # Custom text file
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, 'r') as f:
            text = f.read()
        result = evaluate_tokenizer(tokenizer, text, os.path.basename(args.text_file))
        results.append(result)
    
    # Summary
    print0("\n" + "=" * 50)
    print0("Summary")
    print0("=" * 50)
    
    avg_compression = sum(r['compression'] for r in results) / len(results)
    total_tokens = sum(r['tokens'] for r in results)
    total_chars = sum(r['chars'] for r in results)
    
    print0(f"Average compression: {avg_compression:.2f} chars/token")
    print0(f"Total: {total_chars:,} chars → {total_tokens:,} tokens")


if __name__ == "__main__":
    main()
