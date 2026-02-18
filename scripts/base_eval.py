"""
Base model evaluation script for nanollama.

Usage:
    python -m scripts.base_eval --model-tag=base
"""

import argparse
import json

import torch

from nanollama.common import compute_init, autodetect_device_type, print0
from nanollama.checkpoint_manager import load_model
from nanollama.core_eval import run_all_evals


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nanollama base model")
    parser.add_argument("--model-tag", type=str, default="base", help="Model to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    print0(f"Loading model: {args.model_tag}")
    model, tokenizer, meta = load_model(args.model_tag, device, phase="eval")
    
    print0(f"Model loaded from step {meta['step']}")
    
    # Run evaluations
    print0("\nRunning evaluations...")
    results = run_all_evals(model, tokenizer, device)
    
    # Print results
    print0("\n" + "=" * 60)
    print0("Evaluation Results")
    print0("=" * 60)
    
    if 'core' in results:
        core = results['core']
        print0(f"  CORE Score: {core['core_score']:.4f}")
        print0(f"  BPB: {core['bpb']:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print0(f"\nResults saved to {args.output}")
    
    print0("\nDone!")


if __name__ == "__main__":
    main()
