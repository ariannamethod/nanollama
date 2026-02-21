"""
Base model training script for nanollama.

Train Llama 3 models from scratch with optional personality injection.

Usage:
    python -m scripts.base_train --depth=12

Distributed:
    torchrun --nproc_per_node=8 -m scripts.base_train --depth=24
"""

import os
import sys
import time
import math
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanollama.llama import Llama, LlamaConfig, get_config_for_depth, get_named_config, NAMED_CONFIGS
from nanollama.common import (
    compute_init, compute_cleanup, get_dist_info, print0, print_banner,
    autodetect_device_type, get_peak_flops, DummyWandb
)
from nanollama.dataloader import DistributedDataLoader
from nanollama.checkpoint_manager import save_checkpoint
from nanollama.tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train nanollama base model")
    
    # Model
    parser.add_argument("--model-size", type=str, default=None,
                       choices=list(NAMED_CONFIGS.keys()),
                       help="Named model size (overrides --depth)")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    # nanochat extensions (off by default = standard Llama 3, full llama.cpp compatibility)
    parser.add_argument("--use-post-emb-norm", action="store_true", help="RMSNorm after embedding (nanochat)")
    parser.add_argument("--use-resformer", action="store_true", help="ResFormer per-layer scaling (nanochat)")
    parser.add_argument("--softcap", type=float, default=0.0, help="Logit softcap (0=off, 15=nanochat)")

    # Data
    parser.add_argument("--data-dir", type=str, default=None, help="Training data directory")
    parser.add_argument("--personality-dir", type=str, default=None, help="Personality data directory")
    parser.add_argument("--personality-ratio", type=float, default=0.0, 
                       help="Ratio of personality data in each batch (0.0 to 1.0)")
    
    # Training
    parser.add_argument("--total-batch-size", type=int, default=524288, help="Total batch size in tokens")
    parser.add_argument("--device-batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--num-iterations", type=int, default=-1, help="Training iterations (-1 = auto from Chinchilla ratio)")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (auto if None)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Muon matrix groups")
    
    # Logging & Checkpoints
    parser.add_argument("--run", type=str, default="nanollama", help="Run name for wandb")
    parser.add_argument("--model-tag", type=str, default="base", help="Model tag for checkpoints")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N iterations")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N iterations")
    parser.add_argument("--sample-every", type=int, default=500, help="Generate samples every N iterations")
    parser.add_argument("--core-metric-every", type=int, default=-1, help="CORE eval every N iterations")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    
    return parser.parse_args()


def get_lr_schedule(step, warmup_iters, max_iters, max_lr, min_lr_ratio=0.0):
    """WSD (Warmup-Stable-Decay) schedule. Better than cosine: no need to know total steps upfront."""
    decay_start = int(max_iters * 0.50)  # Last 50% is warmdown (nanochat convention)
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    elif step < decay_start:
        return max_lr  # Stable phase
    else:
        # Linear decay to zero
        progress = (step - decay_start) / (max_iters - decay_start)
        final_lr = max_lr * min_lr_ratio
        return final_lr + (max_lr - final_lr) * (1 - progress)


def main():
    args = parse_args()
    
    # Initialize compute
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    
    if rank == 0:
        print_banner()
    
    # Model config
    if args.model_size:
        config = get_named_config(args.model_size)
        print0(f"\nUsing named config: {args.model_size}")
    else:
        config = get_config_for_depth(args.depth)
    config.vocab_size = args.vocab_size
    config.sequence_len = args.max_seq_len
    config.use_post_emb_norm = args.use_post_emb_norm
    config.use_resformer = args.use_resformer
    config.softcap = args.softcap

    print0(f"\n{'='*60}")
    print0(f"nanollama training - {args.model_size or f'depth={args.depth}'}")
    tied_str = ", tied" if config.tie_embeddings else ""
    ext = []
    if config.use_post_emb_norm: ext.append("post-emb-norm")
    if config.use_resformer: ext.append("resformer")
    if config.softcap > 0: ext.append(f"softcap={config.softcap}")
    ext_str = f", extensions: {', '.join(ext)}" if ext else ", llama.cpp compatible"
    print0(f"Model: {config.n_layer} layers, {config.n_embd} dim, {config.n_head} heads, {config.n_kv_head} KV heads{tied_str}{ext_str}")
    print0(f"{'='*60}\n")
    
    # Create model
    print0("Creating model...")
    with torch.device('meta'):
        model = Llama(config)
    
    # Materialize on device
    model.to_empty(device=device)
    model.init_weights()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Auto iterations from Chinchilla ratio (~10x data:param)
    if args.num_iterations == -1:
        target_tokens = 10 * num_params
        args.num_iterations = max(1000, target_tokens // args.total_batch_size)
        print0(f"Auto iterations: {args.num_iterations} (10x Chinchilla, {target_tokens/1e9:.1f}B tokens)")

    # Compile model
    if device_type == "cuda":
        print0("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Setup optimizer
    print0("Setting up optimizer...")
    optimizer = model.setup_optimizer(weight_decay=args.weight_decay)
    
    # Setup data loader
    print0("Setting up data loader...")
    if args.data_dir is None:
        from nanollama.common import get_base_dir
        args.data_dir = os.path.join(get_base_dir(), "data", "fineweb")
    
    # Calculate gradient accumulation
    tokens_per_batch = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % (tokens_per_batch * world_size) == 0
    grad_accum_steps = args.total_batch_size // (tokens_per_batch * world_size)
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    
    try:
        data_loader = DistributedDataLoader(
            data_dir=args.data_dir,
            sequence_length=args.max_seq_len,
            batch_size=args.device_batch_size,
            rank=rank,
            world_size=world_size,
            personality_dir=args.personality_dir,
            personality_ratio=args.personality_ratio,
        )
    except Exception as e:
        print0(f"Warning: Could not create data loader: {e}")
        print0("Using dummy data for testing...")
        data_loader = None
    
    # Auto LR from model dimension
    if args.lr is None:
        args.lr = 0.02  # fixed base LR; per-group scaling handled in setup_optimizer
    print0(f"Learning rate: {args.lr}")
    
    # Wandb
    if args.wandb and rank == 0:
        import wandb
        wandb.init(project="nanollama", name=args.run, config=vars(args))
        logger = wandb
    else:
        logger = DummyWandb()
    
    # Training loop
    print0("\nStarting training...")
    model.train()
    
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    
    # MFU calculation
    peak_flops = get_peak_flops(torch.cuda.get_device_name() if device_type == "cuda" else "cpu")
    model_flops = model.estimate_flops()
    
    t0 = time.time()
    total_tokens = 0
    
    for step in range(args.num_iterations):
        # Learning rate schedule
        lr = get_lr_schedule(step, args.warmup_iters, args.num_iterations, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (param_group.get('initial_lr', lr) / args.lr)
        
        # Gradient accumulation
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            # Get batch
            if data_loader:
                x, y = data_loader.next_batch()
            else:
                # Dummy data for testing
                x = torch.randint(0, config.vocab_size, (args.device_batch_size, args.max_seq_len))
                y = torch.randint(0, config.vocab_size, (args.device_batch_size, args.max_seq_len))
            
            x, y = x.to(device), y.to(device)
            
            # Forward
            with autocast_ctx:
                loss = model(x, targets=y)
                loss = loss / grad_accum_steps
            
            # Backward
            loss.backward()
            loss_accum += loss.item()
            total_tokens += x.numel()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if step % args.log_every == 0 and rank == 0:
            dt = time.time() - t0
            tok_per_sec = total_tokens / dt if dt > 0 else 0
            mfu = model_flops * tok_per_sec / peak_flops * 100 if peak_flops < float('inf') else 0
            
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s | {mfu:.1f}% MFU")
            
            logger.log({
                'train/loss': loss_accum,
                'train/lr': lr,
                'train/tok_per_sec': tok_per_sec,
                'train/mfu': mfu,
                'train/total_tokens': total_tokens,
            })
        
        # Save checkpoint
        if args.save_every > 0 and step > 0 and step % args.save_every == 0 and rank == 0:
            from nanollama.common import get_base_dir
            checkpoint_dir = os.path.join(get_base_dir(), "checkpoints", args.model_tag)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                config=vars(config),
                checkpoint_dir=checkpoint_dir,
            )
        
        # Sync across ranks
        if ddp:
            dist.barrier()
    
    # Final save
    if rank == 0:
        from nanollama.common import get_base_dir
        checkpoint_dir = os.path.join(get_base_dir(), "checkpoints", args.model_tag)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=args.num_iterations,
            config=vars(config),
            checkpoint_dir=checkpoint_dir,
        )
    
    # Cleanup
    logger.finish()
    compute_cleanup()
    
    print0("\nTraining complete!")


if __name__ == "__main__":
    main()
