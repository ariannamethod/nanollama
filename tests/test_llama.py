"""
Basic tests for nanollama model.
"""

import pytest
import torch


class TestLlamaModel:
    """Tests for Llama model architecture."""
    
    def test_config_for_depth(self):
        """Test config generation for different depths."""
        from nanollama.llama import get_config_for_depth
        
        # Test nano config
        config = get_config_for_depth(6)
        assert config.n_layer == 6
        assert config.n_embd == 384
        assert config.n_head == 6
        assert config.n_kv_head == 2
        
        # Test micro config
        config = get_config_for_depth(12)
        assert config.n_layer == 12
        assert config.n_embd == 512
        
        # Test mini config
        config = get_config_for_depth(16)
        assert config.n_layer == 16
        assert config.n_embd == 768
    
    def test_model_creation(self):
        """Test model can be created."""
        from nanollama.llama import create_model
        
        model = create_model(depth=6, vocab_size=1000)
        assert model is not None
        assert model.config.n_layer == 6
    
    def test_model_forward(self):
        """Test forward pass."""
        from nanollama.llama import create_model
        
        model = create_model(depth=4, vocab_size=1000, sequence_len=64)
        model.init_weights()
        
        # Create dummy input
        x = torch.randint(0, 1000, (2, 32))
        
        # Forward pass
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_model_training(self):
        """Test training forward pass with loss."""
        from nanollama.llama import create_model
        
        model = create_model(depth=4, vocab_size=1000, sequence_len=64)
        model.init_weights()
        
        # Create dummy input and targets
        x = torch.randint(0, 1000, (2, 32))
        y = torch.randint(0, 1000, (2, 32))
        
        # Forward pass with loss
        loss = model(x, targets=y)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Positive loss
    
    def test_gqa_shapes(self):
        """Test GQA shapes are correct."""
        from nanollama.llama import LlamaConfig, Llama
        
        config = LlamaConfig(
            n_layer=2,
            n_embd=64,
            n_head=8,
            n_kv_head=2,  # GQA: 4 query heads per KV head
            vocab_size=100,
            sequence_len=32,
        )
        
        model = Llama(config)
        model.init_weights()
        
        # Check attention layer shapes
        attn = model.layers[0].attn
        
        # Query projection: full heads
        assert attn.c_q.weight.shape == (config.n_head * (config.n_embd // config.n_head), config.n_embd)
        
        # KV projections: fewer heads (GQA)
        head_dim = config.n_embd // config.n_head
        assert attn.c_k.weight.shape == (config.n_kv_head * head_dim, config.n_embd)
        assert attn.c_v.weight.shape == (config.n_kv_head * head_dim, config.n_embd)
    
    def test_swiglu_ffn(self):
        """Test SwiGLU FFN."""
        from nanollama.llama import SwiGLUFFN, LlamaConfig
        
        config = LlamaConfig(n_embd=64, multiple_of=32)
        ffn = SwiGLUFFN(config)
        
        x = torch.randn(2, 16, 64)
        y = ffn(x)
        
        assert y.shape == x.shape
    
    def test_rope_embeddings(self):
        """Test rotary position embeddings."""
        from nanollama.llama import precompute_freqs_cis, apply_rotary_emb
        
        head_dim = 64
        seq_len = 32
        
        cos, sin = precompute_freqs_cis(head_dim, seq_len)
        
        assert cos.shape == (1, seq_len, 1, head_dim // 2)
        assert sin.shape == (1, seq_len, 1, head_dim // 2)
        
        # Test application
        x = torch.randn(2, seq_len, 4, head_dim)  # B, T, H, D
        y = apply_rotary_emb(x, cos, sin)
        
        assert y.shape == x.shape


class TestKVCache:
    """Tests for KV cache."""
    
    def test_kv_cache_creation(self):
        """Test KV cache can be created."""
        from nanollama.engine import KVCache
        
        cache = KVCache(
            batch_size=2,
            num_kv_heads=4,
            seq_len=64,
            head_dim=32,
            num_layers=4,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        
        assert cache.k_cache.shape == (4, 2, 64, 4, 32)
        assert cache.v_cache.shape == (4, 2, 64, 4, 32)
    
    def test_kv_cache_advance(self):
        """Test KV cache position tracking."""
        from nanollama.engine import KVCache
        
        cache = KVCache(
            batch_size=1,
            num_kv_heads=4,
            seq_len=64,
            head_dim=32,
            num_layers=4,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        
        assert cache.get_pos() == 0
        cache.advance(10)
        assert cache.get_pos() == 10
        cache.reset()
        assert cache.get_pos() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
