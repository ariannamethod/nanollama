"""
Supervised fine-tuning (SFT) for nanollama.

Usage:
    python -m scripts.chat_sft
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

Adapted from nanochat for Llama 3 architecture.
"""

import gc
import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanollama.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    get_base_dir, autodetect_device_type, get_peak_flops
)
from nanollama.checkpoint_manager import save_checkpoint, load_model

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="model step to load")
# Training
parser.add_argument("--num-iterations", type=int, default=-1, help="steps (-1 = full epoch)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embeddings")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LR for output")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix params")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR fraction")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="warmup ratio")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="warmdown ratio")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR fraction")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="eval every N steps")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="epochs of MMLU")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="epochs of GSM8K")
args = parser.parse_args()
user_config = vars(args).copy()


def main():
    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float('inf')

    # wandb
    use_dummy_wandb = args.run == "dummy" or not master_process
    if use_dummy_wandb:
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(project="nanollama-sft", name=args.run, config=user_config)

    # Load model
    print0("Loading model...")
    model, tokenizer, meta = load_model("base", device, phase="train")
    orig_model = model
    model = torch.compile(model, dynamic=False)
    
    depth = model.config.n_layer
    num_flops_per_token = model.estimate_flops()
    
    tokens_per_batch = args.device_batch_size * args.max_seq_len
    world_tokens_per_batch = tokens_per_batch * ddp_world_size
    assert args.total_batch_size % world_tokens_per_batch == 0
    grad_accum_steps = args.total_batch_size // world_tokens_per_batch
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    # Initialize optimizer
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    # SFT data mixture
    base_dir = get_base_dir()
    identity_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    
    train_tasks = [
        SmolTalk(split="train"),
    ]
    # Add CustomJSON if file exists
    if os.path.exists(identity_filepath):
        train_tasks.extend([
            CustomJSON(filepath=identity_filepath),
            CustomJSON(filepath=identity_filepath),  # 2 epochs
        ])
    # Add MMLU and GSM8K
    try:
        train_tasks.extend([MMLU(subset="auxiliary_train", split="train") for _ in range(args.mmlu_epochs)])
    except Exception as e:
        print0(f"Warning: Could not load MMLU: {e}")
    try:
        train_tasks.extend([GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)])
    except Exception as e:
        print0(f"Warning: Could not load GSM8K: {e}")
    # Spelling tasks
    train_tasks.extend([
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ])
    
    train_dataset = TaskMixture(train_tasks)
    print0(f"Training mixture: {len(train_dataset):,} rows")

    # LR schedule
    def get_lr_multiplier(progress):
        if progress < args.warmup_ratio:
            return (progress + 1e-8) / args.warmup_ratio
        elif progress <= 1.0 - args.warmdown_ratio:
            return 1.0
        else:
            decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
            return (1 - decay) * 1.0 + decay * args.final_lr_frac

    # Training loop
    print0("Starting SFT training...")
    total_training_time = 0
    step = 0
    last_step = False
    
    # Simple data generator
    cursor = ddp_rank
    
    while True:
        if last_step:
            break
            
        if args.num_iterations > 0 and step >= args.num_iterations:
            last_step = True
        
        # Get batch
        batch_ids = []
        for _ in range(args.device_batch_size):
            conversation = train_dataset[cursor % len(train_dataset)]
            ids, _ = tokenizer.render_conversation(conversation, max_tokens=args.max_seq_len + 1)
            # Pad if needed
            if len(ids) < args.max_seq_len + 1:
                ids = ids + [tokenizer.get_bos_token_id()] * (args.max_seq_len + 1 - len(ids))
            batch_ids.append(ids[:args.max_seq_len + 1])
            cursor += ddp_world_size
        
        batch_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
        inputs = batch_tensor[:, :-1]
        targets = batch_tensor[:, 1:]
        
        # Forward/backward
        t0 = time.time()
        optimizer.zero_grad()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(inputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()
        
        # Optimizer step
        progress = step / max(args.num_iterations, 1) if args.num_iterations > 0 else cursor / len(train_dataset)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        
        dt = time.time() - t0
        total_training_time += dt
        
        # Logging
        if step % 10 == 0:
            tok_per_sec = int(args.total_batch_size / dt) if dt > 0 else 0
            mfu = num_flops_per_token * args.total_batch_size / dt / (gpu_peak_flops * ddp_world_size) * 100 if dt > 0 else 0
            print0(f"step {step:05d} | loss: {loss.item() * grad_accum_steps:.4f} | lrm: {lrm:.2f} | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}%")
            wandb_run.log({
                "step": step,
                "train/loss": loss.item() * grad_accum_steps,
                "train/lrm": lrm,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
            })
        
        step += 1
        
        # Check for epoch completion
        if cursor >= len(train_dataset):
            last_step = True
    
    # Save final checkpoint
    if master_process:
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", args.model_tag or f"d{depth}")
        save_checkpoint(
            orig_model,
            optimizer,
            step,
            {"n_layer": depth, "n_embd": model.config.n_embd},
            checkpoint_dir,
        )
    
    print0(f"SFT complete! Total time: {total_training_time/60:.2f}m")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
