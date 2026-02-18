"""
Dataset utilities for nanollama.
Download and prepare training data.
"""

import os
import json
import numpy as np
from typing import Iterator, Optional, Dict, Any
from nanollama.common import get_base_dir, download_file_with_lock, print0


def download_fineweb_sample(num_shards: int = 10) -> str:
    """
    Download FineWeb sample data for pretraining.
    
    Args:
        num_shards: Number of shards to download
    
    Returns:
        Path to data directory
    """
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "data", "fineweb")
    os.makedirs(data_dir, exist_ok=True)
    
    # FineWeb-Edu sample URLs (placeholder - replace with actual URLs)
    base_url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/sample_{:04d}.bin"
    
    for i in range(num_shards):
        shard_path = os.path.join(data_dir, f"shard_{i:04d}.bin")
        if not os.path.exists(shard_path):
            # Download shard
            url = base_url.format(i)
            try:
                download_file_with_lock(url, f"fineweb/shard_{i:04d}.bin")
            except Exception as e:
                print0(f"Warning: Could not download shard {i}: {e}")
    
    return data_dir


def prepare_personality_data(
    jsonl_path: str,
    output_dir: str,
    tokenizer,
    max_seq_len: int = 2048,
) -> str:
    """
    Prepare personality data for pretraining injection.
    
    Personality data should be JSONL with conversation pairs:
    {"instruction": "...", "response": "..."}
    
    Args:
        jsonl_path: Path to input JSONL file
        output_dir: Directory to save tokenized data
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
    
    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_tokens = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # Format as conversation
            instruction = item.get("instruction", item.get("user", ""))
            response = item.get("response", item.get("assistant", ""))
            
            # Tokenize in chat format
            text = f"<|user_start|>{instruction}<|user_end|><|assistant_start|>{response}<|assistant_end|>"
            tokens = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
            
            # Truncate if needed
            tokens = tokens[:max_seq_len]
            all_tokens.extend(tokens)
    
    # Convert to numpy and save
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    output_path = os.path.join(output_dir, "personality_000.bin")
    tokens_array.tofile(output_path)
    
    print0(f"Saved {len(all_tokens)} personality tokens to {output_path}")
    
    return output_dir


def iterate_documents(data_path: str) -> Iterator[str]:
    """
    Iterate over documents in a dataset file.
    
    Supports:
    - JSONL with "text" field
    - Plain text files
    
    Args:
        data_path: Path to data file
    
    Yields:
        Document texts
    """
    if data_path.endswith('.jsonl') or data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if "text" in item:
                    yield item["text"]
                elif "content" in item:
                    yield item["content"]
    else:
        # Plain text, split by double newlines
        with open(data_path, 'r') as f:
            text = f.read()
            for doc in text.split('\n\n'):
                if doc.strip():
                    yield doc


def tokenize_dataset(
    input_path: str,
    output_dir: str,
    tokenizer,
    shard_size: int = 100_000_000,  # 100M tokens per shard
) -> str:
    """
    Tokenize a dataset and save as binary shards.
    
    Args:
        input_path: Path to input data
        output_dir: Directory for output shards
        tokenizer: Tokenizer to use
        shard_size: Number of tokens per shard
    
    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    current_tokens = []
    shard_idx = 0
    
    for doc in iterate_documents(input_path):
        tokens = tokenizer.encode(doc, prepend=tokenizer.get_bos_token_id())
        current_tokens.extend(tokens)
        
        # Save shard when we have enough tokens
        while len(current_tokens) >= shard_size:
            shard_tokens = current_tokens[:shard_size]
            current_tokens = current_tokens[shard_size:]
            
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
            np.array(shard_tokens, dtype=np.uint16).tofile(shard_path)
            print0(f"Saved shard {shard_idx} with {len(shard_tokens)} tokens")
            shard_idx += 1
    
    # Save remaining tokens
    if current_tokens:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
        np.array(current_tokens, dtype=np.uint16).tofile(shard_path)
        print0(f"Saved shard {shard_idx} with {len(current_tokens)} tokens")
    
    return output_dir


class ConversationDataset:
    """Dataset for SFT with conversation data."""
    
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.conversations = []
        
        # Load conversations
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.conversations.append(item)
        
        print0(f"Loaded {len(self.conversations)} conversations")
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a tokenized conversation."""
        conv = self.conversations[idx]
        
        # Convert to standard format if needed
        if "messages" not in conv:
            # Assume instruction/response format
            conv = {
                "messages": [
                    {"role": "user", "content": conv.get("instruction", conv.get("user", ""))},
                    {"role": "assistant", "content": conv.get("response", conv.get("assistant", ""))},
                ]
            }
        
        # Tokenize
        ids, mask = self.tokenizer.render_conversation(conv, max_tokens=self.max_seq_len)
        
        return {
            "input_ids": ids,
            "labels": ids,
            "attention_mask": mask,
        }
