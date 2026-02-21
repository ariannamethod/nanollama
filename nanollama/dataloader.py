"""
Distributed data loader for nanollama pretraining.
Adapted from nanochat with support for personality data injection.
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple, Iterator, List
from nanollama.common import print0


class DistributedDataLoader:
    """
    Distributed data loader for pretraining.
    
    Features:
    - Sharded data loading across ranks
    - Memory-mapped files for efficiency
    - Resumable from any position
    - Personality data mixing (for nanollama)
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        personality_dir: Optional[str] = None,
        personality_ratio: float = 0.0,
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing tokenized data shards
            sequence_length: Sequence length for training
            batch_size: Per-rank batch size
            rank: Current rank
            world_size: Total number of ranks
            seed: Random seed for shuffling
            personality_dir: Directory containing personality data
            personality_ratio: Ratio of personality data (0.0 to 1.0)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.personality_dir = personality_dir
        self.personality_ratio = personality_ratio
        
        # Find all data shards
        self.shard_files = self._find_shards(data_dir)
        self.num_shards = len(self.shard_files)
        
        if self.num_shards == 0:
            raise ValueError(f"No data shards found in {data_dir}")
        
        print0(f"Found {self.num_shards} data shards")
        
        # Personality data shards
        self.personality_shards = []
        if personality_dir and personality_ratio > 0:
            self.personality_shards = self._find_shards(personality_dir)
            print0(f"Found {len(self.personality_shards)} personality shards")
        
        # State
        self.current_shard_idx = 0
        self.current_position = 0
        self.rng = np.random.default_rng(seed + rank)
        
        # Load first shard (offset by rank for distributed)
        self._load_shard(self.rank % self.num_shards)
    
    def _find_shards(self, directory: str) -> List[str]:
        """Find all .bin shard files in a directory."""
        if not os.path.exists(directory):
            return []
        
        shards = []
        for f in sorted(os.listdir(directory)):
            if f.endswith('.bin'):
                shards.append(os.path.join(directory, f))
        return shards
    
    def _load_shard(self, shard_idx: int):
        """Load a shard into memory-mapped array."""
        shard_path = self.shard_files[shard_idx]
        self.current_data = np.memmap(shard_path, dtype=np.uint16, mode='r')
        self.current_shard_idx = shard_idx
        self.current_position = 0
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next training batch. Base data is read SEQUENTIALLY
        (each token seen once per epoch). Personality data uses random sampling.
        """
        seq_len = self.sequence_length

        # Determine split
        if self.personality_ratio > 0 and len(self.personality_shards) > 0:
            num_personality = int(self.batch_size * self.personality_ratio)
            num_base = self.batch_size - num_personality
        else:
            num_personality = 0
            num_base = self.batch_size

        batch_x = np.zeros((self.batch_size, seq_len), dtype=np.int64)
        batch_y = np.zeros((self.batch_size, seq_len), dtype=np.int64)

        # Base data: sequential streaming (every token seen once)
        for i in range(num_base):
            if self.current_position + seq_len + 1 > len(self.current_data):
                next_shard = (self.current_shard_idx + 1) % self.num_shards
                self._load_shard(next_shard)
            batch_x[i] = self.current_data[self.current_position:self.current_position + seq_len]
            batch_y[i] = self.current_data[self.current_position + 1:self.current_position + seq_len + 1]
            self.current_position += seq_len

        # Personality data: random sampling (intentional for mixing)
        if num_personality > 0:
            shard_idx = self.rng.integers(0, len(self.personality_shards))
            pdata = np.memmap(self.personality_shards[shard_idx], dtype=np.uint16, mode='r')
            for i in range(num_personality):
                j = num_base + i
                max_start = max(1, len(pdata) - seq_len - 1)
                start = self.rng.integers(0, max_start)
                batch_x[j] = pdata[start:start + seq_len]
                batch_y[j] = pdata[start + 1:start + seq_len + 1]
            # Shuffle when mixing personality with base
            perm = self.rng.permutation(self.batch_size)
            batch_x = batch_x[perm]
            batch_y = batch_y[perm]

        return (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y),
        )
    
    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            'shard_idx': self.current_shard_idx,
            'position': self.current_position,
            'rng_state': self.rng.bit_generator.state,
        }
    
    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        self._load_shard(state['shard_idx'])
        self.current_position = state['position']
        self.rng.bit_generator.state = state['rng_state']


class InMemoryDataLoader:
    """Simple in-memory data loader for small datasets."""
    
    def __init__(
        self,
        tokens: np.ndarray,
        sequence_length: int,
        batch_size: int,
        seed: int = 42,
    ):
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch."""
        seq_len = self.sequence_length
        batch_x = np.zeros((self.batch_size, seq_len), dtype=np.int64)
        batch_y = np.zeros((self.batch_size, seq_len), dtype=np.int64)
        
        max_start = len(self.tokens) - seq_len - 1
        
        for i in range(self.batch_size):
            start = self.rng.integers(0, max_start)
            batch_x[i] = self.tokens[start:start + seq_len]
            batch_y[i] = self.tokens[start + 1:start + seq_len + 1]
        
        return (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y),
        )
