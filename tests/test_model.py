"""
Unit tests for nanollama model.

Run with: pytest tests/test_model.py -v

These tests verify:
1. Forward pass produces correct output shapes
2. RoPE frequencies are computed correctly (theta=500000 for Llama 3)
3. GQA attention output shapes are correct
4. Parameter counts match expected values for each config
5. Personality dataloader mixes at the correct ratio
"""

import pytest
import torch
import numpy as np
import math
import tempfile
import os

from nanollama.llama import (
    LlamaConfig, 
    Llama, 
    create_model, 
    get_config_for_depth,
    precompute_freqs_cis,
    CausalSelfAttention,
    SwiGLUFFN,
)


class TestForwardPass:
    """Test that forward pass works correctly."""
    
    def test_forward_shape_nano(self):
        """Test forward pass produces correct output shape for nano config."""
        config = get_config_for_depth(6)
        model = Llama(config)
        
        batch_size, seq_len = 2, 128
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits = model(x)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert logits.dtype == torch.float32  # logits should be float32
    
    def test_forward_with_targets(self):
        """Test forward pass with targets returns loss."""
        config = get_config_for_depth(6)
        model = Llama(config)
        
        batch_size, seq_len = 2, 128
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        loss = model(x, targets=targets)
        
        assert loss.shape == ()  # scalar
        assert loss.dtype == torch.float32
        assert loss.item() > 0  # loss should be positive
    
    def test_create_model_helper(self):
        """Test create_model helper function."""
        model = create_model(depth=6)
        
        assert model.config.n_layer == 6
        assert model.config.n_embd == 384
        assert model.config.n_head == 6
        assert model.config.n_kv_head == 2


class TestRoPE:
    """Test Rotary Position Embeddings."""
    
    def test_rope_theta_llama3(self):
        """Test that RoPE uses theta=500000 (Llama 3), not 10000 (Llama 2)."""
        config = LlamaConfig()
        assert config.rope_theta == 500000.0, \
            f"Llama 3 should use theta=500000, got {config.rope_theta}"
    
    def test_rope_frequency_computation(self):
        """Test RoPE frequencies are computed correctly."""
        dim = 64
        seq_len = 128
        theta = 500000.0  # Llama 3 theta
        
        cos, sin = precompute_freqs_cis(dim, seq_len, theta=theta)
        
        # Check shapes: (1, seq_len, 1, dim//2)
        assert cos.shape == (1, seq_len, 1, dim // 2)
        assert sin.shape == (1, seq_len, 1, dim // 2)
        
        # Verify frequency computation
        # inv_freq = 1 / (theta^(2i/dim)) for i in [0, dim/2)
        expected_inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # At position t, freq = t * inv_freq
        # cos should be cos(t * inv_freq), sin should be sin(t * inv_freq)
        t = 10  # test position
        expected_freqs = t * expected_inv_freq
        
        # Compare (allowing for bf16 precision)
        actual_cos = cos[0, t, 0, :].float()
        expected_cos = torch.cos(expected_freqs)
        
        assert torch.allclose(actual_cos, expected_cos, atol=1e-2), \
            f"RoPE cos mismatch at position {t}"
    
    def test_rope_different_from_llama2(self):
        """Test that Llama 3 frequencies differ from Llama 2."""
        dim = 64
        seq_len = 128
        
        cos_llama3, _ = precompute_freqs_cis(dim, seq_len, theta=500000.0)
        cos_llama2, _ = precompute_freqs_cis(dim, seq_len, theta=10000.0)
        
        # They should be different
        assert not torch.allclose(cos_llama3, cos_llama2), \
            "Llama 3 and Llama 2 RoPE frequencies should differ"


class TestGQA:
    """Test Grouped Query Attention."""
    
    def test_gqa_output_shape(self):
        """Test GQA attention produces correct output shape."""
        config = LlamaConfig(n_layer=1, n_head=12, n_kv_head=4, n_embd=768)
        attn = CausalSelfAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, config.n_embd)
        
        # Prepare RoPE
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, seq_len, theta=config.rope_theta)
        cos_sin = (cos[:, :seq_len], sin[:, :seq_len])
        
        y = attn(x, cos_sin, window_size=(seq_len, 0))
        
        assert y.shape == (batch_size, seq_len, config.n_embd)
    
    def test_gqa_kv_repeat(self):
        """Test that GQA correctly repeats KV heads."""
        config = LlamaConfig(n_layer=1, n_head=12, n_kv_head=4, n_embd=768)
        attn = CausalSelfAttention(config, layer_idx=0)
        
        # n_rep should be n_head / n_kv_head = 12 / 4 = 3
        assert attn.n_rep == 3
        
        # K and V projections should output n_kv_head * head_dim
        head_dim = config.n_embd // config.n_head
        assert attn.c_k.out_features == config.n_kv_head * head_dim
        assert attn.c_v.out_features == config.n_kv_head * head_dim
        
        # Q projection should output n_head * head_dim
        assert attn.c_q.out_features == config.n_head * head_dim


class TestSwiGLU:
    """Test SwiGLU FFN."""
    
    def test_swiglu_output_shape(self):
        """Test SwiGLU produces correct output shape."""
        config = LlamaConfig(n_embd=768)
        ffn = SwiGLUFFN(config)
        
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, config.n_embd)
        
        y = ffn(x)
        
        assert y.shape == (batch_size, seq_len, config.n_embd)
    
    def test_swiglu_hidden_dim(self):
        """Test SwiGLU hidden dimension calculation."""
        config = LlamaConfig(n_embd=768, multiple_of=256)
        ffn = SwiGLUFFN(config)
        
        # Expected: int(2 * (4 * 768) / 3) = 2048, rounded to multiple of 256
        expected_hidden = 2048
        
        assert ffn.gate_proj.out_features == expected_hidden
        assert ffn.up_proj.out_features == expected_hidden
        assert ffn.down_proj.in_features == expected_hidden


class TestParameterCount:
    """Test parameter counts for each configuration."""
    
    @staticmethod
    def calculate_llama_params(config: LlamaConfig, padded_vocab: int = None) -> int:
        """Calculate expected Llama 3 parameter count."""
        if padded_vocab is None:
            padded_vocab = ((config.vocab_size + 63) // 64) * 64
        
        head_dim = config.n_embd // config.n_head
        
        # FFN hidden dim
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # Embeddings (untied)
        embed_params = padded_vocab * config.n_embd  # input embeddings
        unembed_params = padded_vocab * config.n_embd  # output projection
        
        # Per layer
        # Attention: Q, K, V, O projections
        attn_params = (
            config.n_embd * (config.n_head * head_dim) +  # Q
            config.n_embd * (config.n_kv_head * head_dim) +  # K
            config.n_embd * (config.n_kv_head * head_dim) +  # V
            config.n_embd * config.n_embd  # O
        )
        
        # FFN: gate, up, down
        ffn_params = 3 * config.n_embd * hidden_dim
        
        layer_params = attn_params + ffn_params
        total_layer_params = config.n_layer * layer_params
        
        return embed_params + unembed_params + total_layer_params
    
    @pytest.mark.parametrize("depth,expected_approx", [
        (6, 34_000_000),    # ~34M (actual: 34,013,184)
        (12, 69_000_000),   # ~69M (actual: 68,943,872)
        (16, 150_000_000),  # ~150M (actual: 149,815,296)
        (24, 336_000_000),  # ~336M (actual: 336,068,608)
    ])
    def test_param_count_approximation(self, depth, expected_approx):
        """Test parameter counts are in expected ballpark."""
        config = get_config_for_depth(depth)
        model = Llama(config)
        
        actual_params = sum(p.numel() for p in model.parameters())
        
        # Allow 15% variance (vocab padding can affect count slightly)
        lower = expected_approx * 0.85
        upper = expected_approx * 1.15
        
        assert lower < actual_params < upper, \
            f"depth={depth}: expected ~{expected_approx:,}, got {actual_params:,}"
    
    def test_param_count_matches_calculation(self):
        """Test actual param count matches formula."""
        config = get_config_for_depth(6)
        model = Llama(config)
        
        actual = sum(p.numel() for p in model.parameters())
        expected = self.calculate_llama_params(config)
        
        # Should be very close (within 1%)
        assert abs(actual - expected) / expected < 0.01, \
            f"Param count mismatch: expected {expected:,}, got {actual:,}"


class TestDataloader:
    """Test personality dataloader mixing."""
    
    def test_personality_ratio(self):
        """Test that personality_ratio correctly mixes data."""
        from nanollama.dataloader import DistributedDataLoader
        
        # Create temporary data files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base data shard
            base_dir = os.path.join(tmpdir, "base")
            os.makedirs(base_dir)
            base_data = np.zeros(10000, dtype=np.uint16)  # All zeros
            base_data.tofile(os.path.join(base_dir, "shard_0.bin"))
            
            # Create personality data shard
            pers_dir = os.path.join(tmpdir, "personality")
            os.makedirs(pers_dir)
            pers_data = np.ones(10000, dtype=np.uint16)  # All ones
            pers_data.tofile(os.path.join(pers_dir, "shard_0.bin"))
            
            # Test with 20% personality ratio
            batch_size = 100
            personality_ratio = 0.20
            
            loader = DistributedDataLoader(
                data_dir=base_dir,
                sequence_length=32,
                batch_size=batch_size,
                personality_dir=pers_dir,
                personality_ratio=personality_ratio,
            )
            
            # Get a batch
            x, y = loader.next_batch()
            
            assert x.shape == (batch_size, 32)
            
            # Count sequences from each source
            # Base data is all zeros, personality is all ones
            # Due to random sampling within shards, we check the mean
            row_means = x.float().mean(dim=1)
            
            # Rows from personality data should have mean ~1.0
            # Rows from base data should have mean ~0.0
            personality_rows = (row_means > 0.5).sum().item()
            
            # Should be approximately 20% personality
            expected_personality = int(batch_size * personality_ratio)
            
            # Allow 30% variance due to small batch and random sampling
            assert expected_personality * 0.7 <= personality_rows <= expected_personality * 1.3, \
                f"Expected ~{expected_personality} personality rows, got {personality_rows}"


class TestConfigDepth:
    """Test depth-based configuration."""
    
    @pytest.mark.parametrize("depth,n_embd,n_head,n_kv_head", [
        (6, 384, 6, 2),
        (12, 512, 8, 2),
        (16, 768, 12, 4),
        (24, 1024, 16, 4),
        (32, 3200, 32, 8),  # Large uses depth=32 with larger width
    ])
    def test_config_for_depth(self, depth, n_embd, n_head, n_kv_head):
        """Test get_config_for_depth returns expected values."""
        config = get_config_for_depth(depth)
        
        assert config.n_layer == depth
        assert config.n_embd == n_embd
        assert config.n_head == n_head
        assert config.n_kv_head == n_kv_head


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
