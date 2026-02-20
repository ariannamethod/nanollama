#!/usr/bin/env python
"""
Smoke test for nanollama - verifies the entire pipeline works.
Trains nano model for 100 steps on synthetic data and checks that loss decreases.

Usage:
    python -m tests.smoke_test
    
Expected: Completes in ~5 seconds on CPU, ~2 seconds on GPU.
Loss should decrease by at least 10%.

For testing with real data, first run:
    python -m data.prepare_fineweb --num-samples 50000
"""

import os
import sys
import time
import torch

def main():
    print("=" * 60)
    print("nanollama Smoke Test")
    print("=" * 60)
    
    # Import nanollama
    try:
        from nanollama.llama import create_model, LlamaConfig
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
        print(f"✓ Using GPU: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Create nano model (~15M params)
    print("\n--- Creating Model ---")
    config = LlamaConfig(
        sequence_len=512,
        vocab_size=1024,  # Small vocab for test
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
    )
    model = create_model(depth=config.n_layer, 
                         vocab_size=config.vocab_size,
                         n_embd=config.n_embd,
                         n_head=config.n_head,
                         n_kv_head=config.n_kv_head,
                         sequence_len=config.sequence_len)
    model = model.to(device)
    model.init_weights()
    
    nparams = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {nparams:,} parameters")
    
    # Create synthetic data
    print("\n--- Preparing Data ---")
    batch_size = 4
    seq_len = 128
    
    # Random token sequences (simple pattern for quick convergence)
    torch.manual_seed(42)
    data = torch.randint(0, config.vocab_size, (100, seq_len + 1), device=device)
    print(f"✓ Created {len(data)} synthetic sequences")
    
    # Optimizer
    print("\n--- Training ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    num_steps = 100
    losses = []
    t0 = time.time()
    
    model.train()
    for step in range(num_steps):
        # Get batch
        batch_idx = torch.randint(0, len(data), (batch_size,))
        batch = data[batch_idx]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward
        loss = model(inputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0 or step == num_steps - 1:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")
    
    t1 = time.time()
    dt = t1 - t0
    
    # Check results
    print("\n--- Results ---")
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    
    print(f"Initial loss (avg first 5): {initial_loss:.4f}")
    print(f"Final loss (avg last 5):    {final_loss:.4f}")
    print(f"Training time: {dt:.2f}s ({1000*dt/num_steps:.1f}ms/step)")
    
    # Verify loss decreased
    success = final_loss < initial_loss * 0.9  # At least 10% reduction
    
    if success:
        print(f"\n✓ SMOKE TEST PASSED - Loss decreased by {100*(1-final_loss/initial_loss):.1f}%")
    else:
        print(f"\n✗ SMOKE TEST FAILED - Loss did not decrease enough")
    
    # Test generation
    print("\n--- Testing Generation ---")
    model.eval()
    initial_tokens = [0, 1, 2, 3, 4]  # Some token IDs
    generated = []
    for token in model.generate(initial_tokens, max_tokens=10, temperature=0.8):
        generated.append(token)
    print(f"✓ Generated {len(generated)} tokens from initial tokens")
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
