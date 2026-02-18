#!/usr/bin/env python
"""
Integration test for nanollama pipeline.

Tests the complete training → generation → export flow.
Should complete in <60 seconds on CPU.

Usage:
    pytest tests/test_pipeline.py -v
    python -m tests.test_pipeline
"""

import os
import sys
import tempfile
import time
import json

import torch
import numpy as np
import pytest


class TestPipeline:
    """Integration tests for the full nanollama pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic training data (100 samples)."""
        torch.manual_seed(42)
        vocab_size = 512
        seq_len = 64
        n_samples = 100
        
        # Generate random sequences with some structure
        data = torch.randint(0, vocab_size, (n_samples, seq_len))
        return data, vocab_size, seq_len
    
    @pytest.fixture
    def nano_model(self, synthetic_data):
        """Create a nano config model for testing."""
        from nanollama.llama import Llama, LlamaConfig
        
        _, vocab_size, seq_len = synthetic_data
        
        config = LlamaConfig(
            sequence_len=seq_len,
            vocab_size=vocab_size,
            n_layer=2,  # Tiny for fast testing
            n_head=2,
            n_kv_head=1,
            n_embd=64,
        )
        
        model = Llama(config)
        model.init_weights()
        
        return model, config
    
    def test_forward_pass(self, nano_model, synthetic_data):
        """Test that forward pass produces correct output shape."""
        model, config = nano_model
        data, vocab_size, seq_len = synthetic_data
        
        # Get a batch
        batch = data[:4]
        inputs = batch[:, :-1]
        
        # Forward pass
        logits = model(inputs)
        
        assert logits.shape == (4, seq_len - 1, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_training_loss_decreases(self, nano_model, synthetic_data, temp_dir):
        """Test that loss decreases during training (10 steps)."""
        model, config = nano_model
        data, vocab_size, seq_len = synthetic_data
        
        device = torch.device('cpu')
        model = model.to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Training loop
        num_steps = 10
        batch_size = 8
        losses = []
        
        model.train()
        for step in range(num_steps):
            # Get batch
            batch_idx = torch.randint(0, len(data), (batch_size,))
            batch = data[batch_idx].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward + backward
            loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Verify loss decreased
        initial_loss = np.mean(losses[:3])
        final_loss = np.mean(losses[-3:])
        
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    
    def test_generation_produces_tokens(self, nano_model):
        """Test that generation produces non-empty output."""
        model, config = nano_model
        
        model.eval()
        
        # Generate from a prompt
        prompt_tokens = [1, 2, 3, 4, 5]
        generated = list(model.generate(prompt_tokens, max_tokens=10, temperature=0.8))
        
        assert len(generated) > 0, "Generation produced no tokens"
        assert len(generated) == 10, f"Expected 10 tokens, got {len(generated)}"
        
        # Check tokens are in valid range
        for token in generated:
            assert 0 <= token < config.vocab_size, f"Invalid token: {token}"
    
    def test_checkpoint_save_load(self, nano_model, temp_dir):
        """Test saving and loading checkpoints."""
        from nanollama.checkpoint_manager import save_checkpoint, load_checkpoint
        
        model, config = nano_model
        
        # Create dummy optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Save checkpoint
        config_dict = {
            'sequence_len': config.sequence_len,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_kv_head': config.n_kv_head,
            'n_embd': config.n_embd,
        }
        
        save_checkpoint(model, optimizer, step=100, config=config_dict,
                       checkpoint_dir=temp_dir, name="test")
        
        # Find saved checkpoint
        ckpt_path = os.path.join(temp_dir, "test_step100.pt")
        assert os.path.exists(ckpt_path), "Checkpoint not saved"
        
        # Load checkpoint
        ckpt = load_checkpoint(ckpt_path, torch.device('cpu'))
        
        assert 'model_state_dict' in ckpt
        assert 'step' in ckpt
        assert ckpt['step'] == 100
        
        # Verify weights can be loaded
        from nanollama.llama import Llama, LlamaConfig
        new_model = Llama(LlamaConfig(**config_dict))
        new_model.load_state_dict(ckpt['model_state_dict'])
    
    def test_export_gguf_creates_file(self, nano_model, temp_dir):
        """Test that GGUF export creates a non-empty file."""
        from nanollama.checkpoint_manager import save_checkpoint
        
        model, config = nano_model
        
        # Save checkpoint first
        config_dict = {
            'sequence_len': config.sequence_len,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_kv_head': config.n_kv_head,
            'n_embd': config.n_embd,
        }
        
        save_checkpoint(model, None, step=100, config=config_dict,
                       checkpoint_dir=temp_dir, name="test")
        
        ckpt_path = os.path.join(temp_dir, "test_step100.pt")
        output_path = os.path.join(temp_dir, "test.gguf")
        
        # Run export (simplified - just verify the script runs)
        from scripts.export_gguf import (
            write_gguf_header, write_gguf_kv, map_tensor_name,
            GGUF_MAGIC, GGUF_VERSION
        )
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict']
        
        # Verify tensor name mapping works
        for name in state_dict.keys():
            gguf_name = map_tensor_name(name)
            assert gguf_name is not None
            assert len(gguf_name) > 0
        
        # Write a minimal GGUF file to verify the write functions work
        with open(output_path, 'wb') as f:
            write_gguf_header(f, len(state_dict), 5)
            write_gguf_kv(f, "general.architecture", 8, "llama")
        
        assert os.path.exists(output_path), "GGUF file not created"
        assert os.path.getsize(output_path) > 0, "GGUF file is empty"
    
    def test_gamma_extract_inject_roundtrip(self, nano_model, temp_dir):
        """Test gamma extraction and injection."""
        model, config = nano_model
        
        # Create "base" and "personality" checkpoints
        # (just use same model with different random perturbations)
        
        state_base = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Perturb weights to simulate personality training
        state_personality = {}
        for k, v in state_base.items():
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Add small random perturbation
                state_personality[k] = v + 0.01 * torch.randn_like(v)
            else:
                state_personality[k] = v.clone()
        
        # Save checkpoints
        base_path = os.path.join(temp_dir, "base.pt")
        personality_path = os.path.join(temp_dir, "personality.pt")
        gamma_path = os.path.join(temp_dir, "gamma.npz")
        output_path = os.path.join(temp_dir, "injected.pt")
        
        torch.save({'model_state_dict': state_base}, base_path)
        torch.save({'model_state_dict': state_personality}, personality_path)
        
        # Extract gamma
        from scripts.extract_gamma import extract_gamma, save_gamma_npz, load_gamma_npz
        
        gamma = extract_gamma(personality_path, base_path, threshold=1e-8)
        assert len(gamma) > 0, "No gamma extracted"
        
        save_gamma_npz(gamma, gamma_path, sparsity_threshold=1e-10)
        assert os.path.exists(gamma_path), "Gamma file not created"
        
        # Load gamma
        gamma_loaded = load_gamma_npz(gamma_path)
        assert len(gamma_loaded) > 0, "No gamma loaded"
        
        # Inject gamma
        from scripts.inject_gamma import inject_gamma
        
        ckpt, stats = inject_gamma(base_path, gamma_loaded, alpha=1.0)
        assert stats['injected'] > 0, "No layers injected"
        
        # Save and verify
        torch.save(ckpt, output_path)
        assert os.path.exists(output_path), "Injected checkpoint not saved"
        
        # Verify injected weights match personality weights (approximately)
        ckpt_loaded = torch.load(output_path, map_location='cpu', weights_only=False)
        state_injected = ckpt_loaded['model_state_dict']
        
        for key in state_personality:
            if key in state_injected:
                diff = (state_injected[key].float() - state_personality[key].float()).abs().max()
                assert diff < 0.01, f"Injected weights don't match: {key}, diff={diff}"
    
    def test_full_pipeline_under_60_seconds(self, temp_dir):
        """Test the complete pipeline completes in <60 seconds."""
        from nanollama.llama import Llama, LlamaConfig
        from nanollama.checkpoint_manager import save_checkpoint
        
        start_time = time.time()
        
        # 1. Create model
        config = LlamaConfig(
            sequence_len=64,
            vocab_size=512,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
        )
        model = Llama(config)
        model.init_weights()
        
        # 2. Create synthetic data (100 samples)
        torch.manual_seed(42)
        data = torch.randint(0, config.vocab_size, (100, config.sequence_len))
        
        # 3. Train for 10 steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        
        losses = []
        for step in range(10):
            batch_idx = torch.randint(0, len(data), (8,))
            batch = data[batch_idx]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            
            loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        assert losses[-1] < losses[0], "Loss did not decrease"
        
        # 4. Generate
        model.eval()
        generated = list(model.generate([1, 2, 3], max_tokens=5, temperature=0.8))
        assert len(generated) == 5, "Generation failed"
        
        # 5. Save checkpoint
        config_dict = {
            'sequence_len': config.sequence_len,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_kv_head': config.n_kv_head,
            'n_embd': config.n_embd,
        }
        save_checkpoint(model, optimizer, step=10, config=config_dict,
                       checkpoint_dir=temp_dir, name="pipeline")
        
        ckpt_path = os.path.join(temp_dir, "pipeline_step10.pt")
        assert os.path.exists(ckpt_path)
        
        # 6. Export GGUF (just write header to verify)
        from scripts.export_gguf import write_gguf_header
        
        gguf_path = os.path.join(temp_dir, "pipeline.gguf")
        with open(gguf_path, 'wb') as f:
            write_gguf_header(f, 10, 5)
        
        assert os.path.exists(gguf_path)
        assert os.path.getsize(gguf_path) > 0
        
        elapsed = time.time() - start_time
        print(f"\nFull pipeline completed in {elapsed:.2f} seconds")
        
        assert elapsed < 60, f"Pipeline took too long: {elapsed:.2f}s > 60s"


def main():
    """Run tests manually."""
    print("=" * 60)
    print("nanollama Pipeline Integration Tests")
    print("=" * 60)
    
    # Create test instance
    test = TestPipeline()
    
    # Create fixtures
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp dir: {temp_dir}")
        
        torch.manual_seed(42)
        vocab_size = 512
        seq_len = 64
        n_samples = 100
        data = torch.randint(0, vocab_size, (n_samples, seq_len))
        synthetic_data = (data, vocab_size, seq_len)
        
        from nanollama.llama import Llama, LlamaConfig
        config = LlamaConfig(
            sequence_len=seq_len,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=64,
        )
        model = Llama(config)
        model.init_weights()
        nano_model = (model, config)
        
        tests = [
            ("Forward pass", lambda: test.test_forward_pass(nano_model, synthetic_data)),
            ("Training loss decreases", lambda: test.test_training_loss_decreases(nano_model, synthetic_data, temp_dir)),
            ("Generation produces tokens", lambda: test.test_generation_produces_tokens(nano_model)),
            ("Checkpoint save/load", lambda: test.test_checkpoint_save_load(nano_model, temp_dir)),
            ("GGUF export", lambda: test.test_export_gguf_creates_file(nano_model, temp_dir)),
            ("Gamma extract/inject", lambda: test.test_gamma_extract_inject_roundtrip(nano_model, temp_dir)),
            ("Full pipeline <60s", lambda: test.test_full_pipeline_under_60_seconds(temp_dir)),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_fn in tests:
            try:
                print(f"\n--- {name} ---")
                test_fn()
                print(f"✓ {name} PASSED")
                passed += 1
            except Exception as e:
                print(f"✗ {name} FAILED: {e}")
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)
        
        return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
