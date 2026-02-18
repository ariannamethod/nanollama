"""
Reinforcement learning on GSM8K for nanollama.

Simplified GRPO-style RL:
1) No trust region / KL regularization
2) On-policy (no PPO ratio+clip)
3) Token-level normalization
4) Simple advantage: (r - mu)

Usage:
    python -m scripts.chat_rl
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default

Adapted from nanochat for Llama 3 architecture.
"""

import argparse
import os
import itertools
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanollama.common import (
    compute_init, compute_cleanup, print0, get_base_dir,
    DummyWandb, autodetect_device_type
)
from nanollama.checkpoint_manager import save_checkpoint, load_model
from nanollama.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="model step to load")
# Training
parser.add_argument("--num-epochs", type=int, default=1, help="epochs over GSM8K")
parser.add_argument("--device-batch-size", type=int, default=8, help="batch size per forward")
parser.add_argument("--examples-per-step", type=int, default=16, help="examples per step")
parser.add_argument("--num-samples", type=int, default=16, help="samples per example")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embeddings")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LR for output")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix params")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR fraction")
# Evaluation
parser.add_argument("--eval-every", type=int, default=60, help="eval every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="examples for eval")
parser.add_argument("--save-every", type=int, default=60, help="save every N steps")
args = parser.parse_args()
user_config = vars(args).copy()


def main():
    # Init compute
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # wandb
    use_dummy_wandb = args.run == "dummy" or not master_process
    if use_dummy_wandb:
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(project="nanollama-rl", name=args.run, config=user_config)

    # Load model
    print0("Loading SFT model...")
    model, tokenizer, meta = load_model("sft", device, phase="eval")
    engine = Engine(model, tokenizer)

    # Tasks
    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
    print0(f"Calculated number of steps: {num_steps}")

    # Initialize optimizer
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    # LR schedule
    def get_lr_multiplier(it):
        return 1.0 - it / num_steps

    # Examples per rank
    assert args.examples_per_step % ddp_world_size == 0
    examples_per_rank = args.examples_per_step // ddp_world_size
    print0(f"Examples per rank: {examples_per_rank}")

    # Training loop
    print0("Starting RL training...")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    example_iter = itertools.cycle(rank_indices)

    for step in range(num_steps):
        # Evaluation
        if step % args.eval_every == 0:
            model.eval()
            correct = 0
            total = 0
            for idx in range(ddp_rank, min(args.eval_examples, len(val_task)), ddp_world_size):
                conversation = val_task[idx]
                tokens = tokenizer.render_for_completion(conversation)
                with autocast_ctx:
                    results, _ = engine.generate_batch(
                        tokens,
                        num_samples=1,
                        max_tokens=args.max_new_tokens,
                        temperature=0.0,
                        top_k=args.top_k,
                    )
                for result in results:
                    generated = tokenizer.decode(result[len(tokens):])
                    if val_task.evaluate(conversation, generated):
                        correct += 1
                    total += 1
            
            if ddp:
                correct_t = torch.tensor([correct], device=device)
                total_t = torch.tensor([total], device=device)
                dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
                correct = correct_t.item()
                total = total_t.item()
            
            acc = correct / total if total > 0 else 0
            print0(f"Step {step} | Accuracy: {100*acc:.2f}%")
            wandb_run.log({"step": step, "accuracy": acc})

        # Training step
        model.train()
        rewards_list = []
        
        for _ in range(examples_per_rank):
            example_idx = next(example_iter)
            conversation = train_task[example_idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            # Generate samples
            model.eval()
            with autocast_ctx:
                generated_seqs, masks = engine.generate_batch(
                    tokens,
                    num_samples=args.num_samples,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )

            # Calculate rewards
            rewards = []
            for seq in generated_seqs:
                generated = tokenizer.decode(seq[prefix_length:])
                reward = train_task.reward(conversation, generated)
                rewards.append(reward)
            
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
            advantages = rewards_t - rewards_t.mean()
            rewards_list.append(rewards_t.mean().item())

            # Pad sequences
            max_len = max(len(s) for s in generated_seqs)
            # Get padding token (fall back to 0 if encode_special returns None or empty)
            pad_token = tokenizer.encode_special("<|assistant_end|>")
            if not pad_token:  # handles None or empty list
                pad_token = 0
            elif isinstance(pad_token, list):
                pad_token = pad_token[0] if pad_token else 0
            padded_seqs = [s + [pad_token] * (max_len - len(s)) for s in generated_seqs]
            padded_masks = [m + [0] * (max_len - len(m)) for m in masks]

            ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_ids[:, 1:] == 0] = -1

            # PG objective
            model.train()
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
            
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / num_valid
            loss = -pg_obj
            loss.backward()

        # Update
        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)

        mean_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0
        print0(f"Step {step}/{num_steps} | Reward: {mean_reward:.4f} | LRM: {lrm:.2f}")
        wandb_run.log({"step": step, "reward": mean_reward, "lrm": lrm})

        # Save checkpoint
        if master_process and step > 0 and step % args.save_every == 0:
            base_dir = get_base_dir()
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", args.model_tag or f"d{model.config.n_layer}")
            save_checkpoint(
                model,
                None,
                step,
                {"n_layer": model.config.n_layer, "n_embd": model.config.n_embd},
                checkpoint_dir,
            )

    print0("RL training complete!")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
