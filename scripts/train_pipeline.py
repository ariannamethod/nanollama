"""
Automated training pipeline for nanollama.

Full sequence: base → personality → gamma → GGUF export.
One command, no babysitting.

Usage:
    # Full pipeline (base + personality + gamma + GGUF)
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=small \
        --personality-dir=~/.cache/nanollama/data/personality/yent \
        --personality-name=yent

    # Base only
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=mini --base-only

    # Skip GGUF export
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=small --no-gguf \
        --personality-dir=~/.cache/nanollama/data/personality/yent

    # Personality only (base already trained)
    torchrun --nproc_per_node=4 -m scripts.train_pipeline \
        --model-size=small --personality-only \
        --personality-dir=~/.cache/nanollama/data/personality/yent
"""

import os
import sys
import subprocess
import argparse

from nanollama.llama import NAMED_CONFIGS
from nanollama.common import get_base_dir, print0


def parse_args():
    parser = argparse.ArgumentParser(description="nanollama training pipeline")

    # Model — the only required arg
    parser.add_argument("--model-size", type=str, required=True,
                        choices=list(NAMED_CONFIGS.keys()),
                        help="Named model size (nano/micro/mini/small/goldie/...)")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Training data directory (auto-detected if not set)")
    parser.add_argument("--personality-dir", type=str, default=None,
                        help="Personality data directory")
    parser.add_argument("--personality-name", type=str, default=None,
                        help="Personality name for model tag (e.g. 'yent', 'arianna')")
    parser.add_argument("--personality-ratio", type=float, default=0.2,
                        help="Personality data ratio (default: 0.2)")

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

    # Pipeline control — disable steps you don't need
    parser.add_argument("--base-only", action="store_true",
                        help="Only train base, skip personality/gamma/gguf")
    parser.add_argument("--personality-only", action="store_true",
                        help="Skip base (already trained), run personality/gamma/gguf")
    parser.add_argument("--no-gamma", action="store_true",
                        help="Skip gamma extraction")
    parser.add_argument("--no-gguf", action="store_true",
                        help="Skip GGUF export")

    # GGUF export
    parser.add_argument("--gguf-dtype", type=str, default="f16",
                        choices=["f32", "f16", "q8_0"],
                        help="GGUF weight dtype (default: f16)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer .model path (auto-detected if not set)")

    # Extensions (passed through to base_train)
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--use-post-emb-norm", action="store_true")
    parser.add_argument("--use-resformer", action="store_true")
    parser.add_argument("--softcap", type=float, default=0.0)

    return parser.parse_args()


def run_training(args, model_tag, personality_dir=None, personality_ratio=0.0):
    """Launch base_train.py as subprocess."""
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

    cmd += ["--run", f"nanollama-{model_tag}"]

    print0(f"\n{'='*60}")
    print0(f"PIPELINE: Training {model_tag}")
    print0(f"Command: {' '.join(cmd)}")
    print0(f"{'='*60}\n")

    result = subprocess.run(cmd, env=os.environ)
    if result.returncode != 0:
        print0(f"\nERROR: Training {model_tag} failed (exit {result.returncode})")
        sys.exit(result.returncode)

    print0(f"\nPIPELINE: {model_tag} complete!")


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

    for path, label in [(base_ckpt, "Base"), (personality_ckpt, "Personality")]:
        if not os.path.exists(path):
            print0(f"ERROR: {label} checkpoint not found: {path}")
            sys.exit(1)

    cmd = [
        sys.executable, "-m", "scripts.extract_gamma",
        "--personality_ckpt", personality_ckpt,
        "--base_ckpt", base_ckpt,
        "--output", output,
    ]

    print0(f"\n{'='*60}")
    print0(f"PIPELINE: Extracting gamma")
    print0(f"  {personality_tag} - {base_tag} -> {os.path.basename(output)}")
    print0(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print0(f"\nERROR: Gamma extraction failed (exit {result.returncode})")
        sys.exit(result.returncode)

    print0(f"\nPIPELINE: Gamma -> {output}")


def run_gguf_export(args, model_tag):
    """Export checkpoint to GGUF format."""
    base_dir = get_base_dir()
    checkpoint = os.path.join(
        base_dir, "checkpoints", model_tag,
        f"checkpoint_step{args.num_iterations}.pt"
    )

    if not os.path.exists(checkpoint):
        print0(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Auto-detect tokenizer
    tokenizer = args.tokenizer
    if not tokenizer:
        tokenizer = os.path.join(base_dir, "tokenizer", "tokenizer.model")
        if not os.path.exists(tokenizer):
            tokenizer = None

    output = os.path.join(
        base_dir, "checkpoints", model_tag,
        f"{model_tag}-{args.gguf_dtype}.gguf"
    )

    cmd = [
        sys.executable, "-m", "scripts.export_gguf",
        "--checkpoint", checkpoint,
        "--output", output,
        "--dtype", args.gguf_dtype,
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]

    print0(f"\n{'='*60}")
    print0(f"PIPELINE: Exporting GGUF ({args.gguf_dtype})")
    print0(f"  {checkpoint} -> {os.path.basename(output)}")
    print0(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print0(f"\nERROR: GGUF export failed (exit {result.returncode})")
        sys.exit(result.returncode)

    print0(f"\nPIPELINE: GGUF -> {output}")


def main():
    args = parse_args()

    size = args.model_size
    base_tag = f"{size}-base"

    # Determine personality tag
    if args.personality_name:
        personality_tag = f"{size}-{args.personality_name}"
    elif args.personality_dir:
        personality_tag = f"{size}-{os.path.basename(args.personality_dir.rstrip('/'))}"
    else:
        personality_tag = None

    # Validate
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
    print0(f"Steps: {args.num_iterations}")

    step_num = 0
    steps = []
    if not args.personality_only:
        step_num += 1
        steps.append(f"{step_num}. Train base -> {base_tag}")
    if has_personality and not args.base_only:
        step_num += 1
        steps.append(f"{step_num}. Train personality -> {personality_tag}")
        if not args.no_gamma:
            step_num += 1
            steps.append(f"{step_num}. Extract gamma")
    if not args.no_gguf:
        # Export whichever models were trained
        if not args.personality_only:
            step_num += 1
            steps.append(f"{step_num}. Export GGUF -> {base_tag}-{args.gguf_dtype}.gguf")
        if has_personality and not args.base_only:
            step_num += 1
            steps.append(f"{step_num}. Export GGUF -> {personality_tag}-{args.gguf_dtype}.gguf")

    for s in steps:
        print0(f"  {s}")
    print0()

    # === Execute pipeline ===

    # Base training
    if not args.personality_only:
        run_training(args, base_tag)

    # Personality training
    if has_personality and not args.base_only:
        run_training(
            args, personality_tag,
            personality_dir=args.personality_dir,
            personality_ratio=args.personality_ratio,
        )
        # Gamma extraction
        if not args.no_gamma:
            run_gamma_extraction(args, base_tag, personality_tag)

    # GGUF export
    if not args.no_gguf:
        if not args.personality_only:
            run_gguf_export(args, base_tag)
        if has_personality and not args.base_only:
            run_gguf_export(args, personality_tag)

    print0(f"\n{'='*60}")
    print0("PIPELINE COMPLETE!")
    print0(f"{'='*60}")


if __name__ == "__main__":
    main()
