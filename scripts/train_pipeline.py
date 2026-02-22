"""
Automated training pipeline for nanollama.

Runs the full sequence: base → personality → gamma extraction.
Single command, no babysitting required.

Usage:
    # Full pipeline: base + personality + gamma
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=small \
        --personality-dir=~/.cache/nanollama/data/personality/yent \
        --personality-name=yent

    # Base only (skip personality + gamma)
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=small --base-only

    # Just personality + gamma (base already trained)
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=small \
        --personality-dir=~/.cache/nanollama/data/personality/yent \
        --personality-name=yent --personality-only
"""

import os
import sys
import subprocess
import argparse

from nanollama.llama import NAMED_CONFIGS
from nanollama.common import get_base_dir, print0, get_dist_info


def parse_args():
    parser = argparse.ArgumentParser(description="nanollama training pipeline")

    # Model
    parser.add_argument("--model-size", type=str, required=True,
                        choices=list(NAMED_CONFIGS.keys()),
                        help="Named model size")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Training data directory")
    parser.add_argument("--personality-dir", type=str, default=None,
                        help="Personality data directory")
    parser.add_argument("--personality-name", type=str, default=None,
                        help="Personality name (used for model tag, e.g. 'yent', 'arianna')")
    parser.add_argument("--personality-ratio", type=float, default=0.2,
                        help="Ratio of personality data in each batch (default: 0.2)")

    # Training
    parser.add_argument("--num-iterations", type=int, default=5000,
                        help="Training iterations (SAME for base and personality)")
    parser.add_argument("--total-batch-size", type=int, default=524288,
                        help="Total batch size in tokens")
    parser.add_argument("--device-batch-size", type=int, default=None,
                        help="Per-device batch size (auto if None)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N iterations")

    # Pipeline control
    parser.add_argument("--base-only", action="store_true",
                        help="Only train base model, skip personality + gamma")
    parser.add_argument("--personality-only", action="store_true",
                        help="Skip base (already trained), only personality + gamma")
    parser.add_argument("--no-gamma", action="store_true",
                        help="Skip gamma extraction after personality training")

    # Extensions (passed through to base_train)
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--use-post-emb-norm", action="store_true")
    parser.add_argument("--use-resformer", action="store_true")
    parser.add_argument("--softcap", type=float, default=0.0)

    return parser.parse_args()


def run_training(args, model_tag, personality_dir=None, personality_ratio=0.0):
    """Launch base_train.py as subprocess with the given config."""
    cmd = [sys.executable, "-m", "scripts.base_train"]

    cmd += ["--model-size", args.model_size]
    cmd += ["--model-tag", model_tag]
    cmd += ["--num-iterations", str(args.num_iterations)]
    cmd += ["--total-batch-size", str(args.total_batch_size)]
    cmd += ["--save-every", str(args.save_every)]
    cmd += ["--log-every", str(args.log_every)]

    if args.device_batch_size:
        cmd += ["--device-batch-size", str(args.device_batch_size)]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if personality_dir:
        cmd += ["--personality-dir", personality_dir]
        cmd += ["--personality-ratio", str(personality_ratio)]
    if args.wandb:
        cmd += ["--wandb"]
    if args.use_qk_norm:
        cmd += ["--use-qk-norm"]
    if args.use_post_emb_norm:
        cmd += ["--use-post-emb-norm"]
    if args.use_resformer:
        cmd += ["--use-resformer"]
    if args.softcap > 0:
        cmd += ["--softcap", str(args.softcap)]

    # Run name for wandb
    cmd += ["--run", f"nanollama-{model_tag}"]

    print0(f"\n{'='*60}")
    print0(f"PIPELINE: Training {model_tag}")
    print0(f"Command: {' '.join(cmd)}")
    print0(f"{'='*60}\n")

    result = subprocess.run(cmd, env=os.environ)
    if result.returncode != 0:
        print0(f"\nERROR: Training {model_tag} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print0(f"\nPIPELINE: {model_tag} training complete!")


def run_gamma_extraction(args, base_tag, personality_tag):
    """Extract gamma = personality - base."""
    base_dir = get_base_dir()
    base_ckpt = os.path.join(
        base_dir, "checkpoints", base_tag,
        f"checkpoint_step{args.num_iterations}.pt"
    )
    personality_ckpt = os.path.join(
        base_dir, "checkpoints", personality_tag,
        f"checkpoint_step{args.num_iterations}.pt"
    )
    output = os.path.join(
        base_dir, "checkpoints", personality_tag,
        f"gamma_{personality_tag}.npz"
    )

    if not os.path.exists(base_ckpt):
        print0(f"ERROR: Base checkpoint not found: {base_ckpt}")
        sys.exit(1)
    if not os.path.exists(personality_ckpt):
        print0(f"ERROR: Personality checkpoint not found: {personality_ckpt}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "scripts.extract_gamma",
        "--personality_ckpt", personality_ckpt,
        "--base_ckpt", base_ckpt,
        "--output", output,
    ]

    print0(f"\n{'='*60}")
    print0(f"PIPELINE: Extracting gamma")
    print0(f"  base: {base_ckpt}")
    print0(f"  personality: {personality_ckpt}")
    print0(f"  output: {output}")
    print0(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print0(f"\nERROR: Gamma extraction failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print0(f"\nPIPELINE: Gamma extracted to {output}")


def main():
    args = parse_args()

    size = args.model_size
    base_tag = f"{size}-base"

    # Determine personality tag
    if args.personality_name:
        personality_tag = f"{size}-{args.personality_name}"
    elif args.personality_dir:
        # Infer name from directory
        personality_tag = f"{size}-{os.path.basename(args.personality_dir.rstrip('/'))}"
    else:
        personality_tag = None

    # Validate flags
    has_personality = args.personality_dir is not None
    if not has_personality and not args.base_only:
        print0("No --personality-dir specified. Running base only.")
        args.base_only = True

    if args.personality_only and not has_personality:
        print0("ERROR: --personality-only requires --personality-dir")
        sys.exit(1)

    # Print plan
    print0("=" * 60)
    print0("nanollama Training Pipeline")
    print0("=" * 60)
    print0(f"Model: {size}")
    print0(f"Steps: {args.num_iterations} (same for base and personality)")

    steps = []
    if not args.personality_only:
        steps.append(f"1. Train base -> {base_tag}")
    if has_personality and not args.base_only:
        step_num = 1 if args.personality_only else 2
        steps.append(f"{step_num}. Train personality -> {personality_tag}")
        if not args.no_gamma:
            steps.append(f"{step_num + 1}. Extract gamma -> gamma_{personality_tag}.npz")

    for s in steps:
        print0(f"  {s}")
    print0()

    # Step 1: Base training
    if not args.personality_only:
        run_training(args, base_tag)

    # Step 2: Personality training
    if has_personality and not args.base_only:
        run_training(
            args, personality_tag,
            personality_dir=args.personality_dir,
            personality_ratio=args.personality_ratio,
        )

        # Step 3: Gamma extraction (only on rank 0 for DDP)
        if not args.no_gamma:
            run_gamma_extraction(args, base_tag, personality_tag)

    print0(f"\n{'='*60}")
    print0("PIPELINE COMPLETE!")
    print0(f"{'='*60}")


if __name__ == "__main__":
    main()
