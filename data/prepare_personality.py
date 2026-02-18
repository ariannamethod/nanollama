"""
Prepare personality data for nanollama pretraining injection.

Usage:
    python -m data.prepare_personality --input=conversations.jsonl
"""

import os
import json
import argparse
import numpy as np

from nanollama.common import get_base_dir, print0


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare personality data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(get_base_dir(), "data", "personality")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print0(f"Preparing personality data from {args.input}")
    
    # Load tokenizer
    from nanollama.tokenizer import get_tokenizer
    try:
        tokenizer = get_tokenizer()
    except Exception as e:
        print0(f"Warning: Could not load tokenizer: {e}")
        print0("Please train a tokenizer first with: python -m scripts.tok_train")
        return
    
    # Process conversations
    all_tokens = []
    num_conversations = 0
    
    with open(args.input, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Support multiple formats
            if "messages" in item:
                # Standard chat format
                conversation = item
            elif "instruction" in item and "response" in item:
                # Instruction/response format
                conversation = {
                    "messages": [
                        {"role": "user", "content": item["instruction"]},
                        {"role": "assistant", "content": item["response"]},
                    ]
                }
            elif "user" in item and "assistant" in item:
                # Simple user/assistant format
                conversation = {
                    "messages": [
                        {"role": "user", "content": item["user"]},
                        {"role": "assistant", "content": item["assistant"]},
                    ]
                }
            else:
                continue
            
            # Tokenize
            try:
                ids, _ = tokenizer.render_conversation(conversation, max_tokens=args.max_seq_len)
                all_tokens.extend(ids)
                num_conversations += 1
            except Exception as e:
                print0(f"Warning: Failed to tokenize conversation: {e}")
                continue
    
    # Save as binary
    output_path = os.path.join(args.output_dir, "personality_000.bin")
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    tokens_array.tofile(output_path)
    
    print0(f"\nPrepared personality data:")
    print0(f"  Conversations: {num_conversations}")
    print0(f"  Tokens: {len(all_tokens):,}")
    print0(f"  Output: {output_path}")
    print0("\nTo use during training, add:")
    print0(f"  --personality-dir={args.output_dir} --personality-ratio=0.2")


if __name__ == "__main__":
    main()
