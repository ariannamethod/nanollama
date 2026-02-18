"""
Download and prepare Shakespeare dataset for quick smoke tests.
The entire pipeline can be verified in 5 minutes using this dataset.

Usage:
    python -m data.prepare_shakespeare
"""

import json
import os
import argparse

from nanollama.common import download_file_with_lock, get_base_dir


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_shakespeare(output_dir: str = None):
    """Download Shakespeare text and prepare for training."""
    if output_dir is None:
        output_dir = os.path.join(get_base_dir(), "data", "shakespeare")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download
    print(f"Downloading Shakespeare from {SHAKESPEARE_URL}...")
    text_path = download_file_with_lock(SHAKESPEARE_URL, "shakespeare.txt")
    
    with open(text_path, 'r') as f:
        text = f.read()
    
    print(f"Downloaded {len(text):,} characters")
    
    # Split into train/val (90/10)
    n = len(text)
    train_text = text[:int(0.9 * n)]
    val_text = text[int(0.9 * n):]
    
    # Save as text files
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    
    with open(train_path, 'w') as f:
        f.write(train_text)
    
    with open(val_path, 'w') as f:
        f.write(val_text)
    
    print(f"Train: {len(train_text):,} chars -> {train_path}")
    print(f"Val: {len(val_text):,} chars -> {val_path}")
    
    # Also create JSONL format for personality-style training
    jsonl_path = os.path.join(output_dir, "shakespeare.jsonl")
    
    # Split into chunks and create conversation-style data
    chunk_size = 500
    with open(jsonl_path, 'w') as f:
        for i in range(0, len(train_text) - chunk_size, chunk_size):
            chunk = train_text[i:i + chunk_size]
            # Create simple prompt-completion pair
            msg = [
                {"role": "user", "content": "Continue this Shakespeare passage:"},
                {"role": "assistant", "content": chunk}
            ]
            f.write(json.dumps(msg) + "\n")
    
    print(f"JSONL: {jsonl_path}")
    print("Done! Use --data-dir for training.")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Shakespeare dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    prepare_shakespeare(args.output_dir)
