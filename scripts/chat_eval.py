"""
Evaluate the Chat model on various benchmarks.

Usage:
    python -m scripts.chat_eval -i sft -a ARC-Easy
    torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC-Easy

Adapted from nanochat for Llama 3 architecture.
"""

import argparse
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanollama.common import (
    compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
)
from nanollama.checkpoint_manager import load_model
from nanollama.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


def run_generative_eval(task_object, tokenizer, model, engine,
                        num_samples, max_new_tokens, temperature, top_k, max_problems=None):
    """Generative evaluation: sample and evaluate."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(r[prefix_length:]) for r in results]
        outcomes = [task_object.evaluate(conversation, c) for c in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

    if ddp:
        num_passed_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        num_passed = num_passed_t.item()
        total = total_t.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    return num_passed / total


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    """Categorical evaluation: check logits for correct answer."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(c) for c in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompts = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(prompt_tensor)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded = tokenizer.encode(letter)
                    assert len(encoded) == 1
                    letter_to_id_cache[letter] = encoded[0]
                letter_ids.append(letter_to_id_cache[letter])
            
            answer_pos = answer_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_idx = focus_logits.argmax(dim=-1).item()
            predicted = letters[argmax_idx]
            outcome = task_object.evaluate(conversation, predicted)
            num_passed += int(outcome)
            total += 1

    if ddp:
        num_passed_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        num_passed = num_passed_t.item()
        total = total_t.item()

    avg = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100*avg:.2f}%)")
    return avg


def run_chat_eval(task_name, model, tokenizer, engine,
                  batch_size=1, num_samples=1, max_new_tokens=512,
                  temperature=0.0, top_k=50, max_problems=None):
    """Run evaluation for a specific task."""
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine,
                                  num_samples, max_new_tokens, temperature, top_k, max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems)
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name or all")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-g', '--model-tag', type=str, default=None)
    parser.add_argument('-s', '--step', type=int, default=None)
    parser.add_argument('-x', '--max-problems', type=int, default=None)
    parser.add_argument('--device-type', type=str, default='')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval")
    engine = Engine(model, tokenizer)

    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
        'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    results = {}
    for task_name in task_names:
        print0(f"\nEvaluating {task_name}...")
        with autocast_ctx:
            acc = run_chat_eval(
                task_name, model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # ChatCORE metric
    if all(t in results for t in all_tasks):
        centered_mean = sum(
            (results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t])
            for t in all_tasks
        ) / len(all_tasks)
        print0(f"\nChatCORE metric: {centered_mean:.4f}")

    compute_cleanup()
