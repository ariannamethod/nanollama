#!/bin/bash
# =============================================================================
# nanollama — Universal Training Pipeline
#
# One command to train any Llama 3 model from scratch on Lambda Cloud.
# Handles: data prep → base training → personality training → gamma → GGUF.
#
# Usage:
#   bash runs/lambda_train.sh --name mini
#   bash runs/lambda_train.sh --name mini --personality data.jsonl
#   bash runs/lambda_train.sh --name small --steps 15000
#   bash runs/lambda_train.sh --name nano --base-only
#   bash runs/lambda_train.sh --name mini --corpus fineweb   # override corpus
#
# Model sizes:
#   nano(34M)  micro(69M)  mini(150M)  small(336M)  goldie(1.1B)  medium(1.6B)  large(3.7B)  big(7.0B)
#
# Data corpus:
#   nano/micro  → FineWeb-Edu only (small models, simple data)
#   mini+       → Multi-corpus SmolLM2 recipe:
#                  FineWeb-Edu 55%, DCLM 25%, Stack v2 10%, MegaMath 10%
#
# =============================================================================

set -e

# ---- Size configs ----
#                        DEPTH    STEPS     BATCH       PARAMS
declare -A CFG_DEPTH=(   [nano]=6    [micro]=12   [mini]=16    [small]=24   [goldie]=20    [medium]=28    [large]=32   [big]=36     )
declare -A CFG_STEPS=(   [nano]=5000 [micro]=-1 [mini]=-1 [small]=-1 [goldie]=-1 [medium]=-1 [large]=-1 [big]=-1   )
declare -A CFG_BATCH=(   [nano]=131072 [micro]=262144 [mini]=262144 [small]=524288 [goldie]=524288 [medium]=1048576 [large]=1048576 [big]=4194304 )
declare -A CFG_PARAMS=(  [nano]="34M" [micro]="69M" [mini]="150M" [small]="336M" [goldie]="1.1B" [medium]="1.6B" [large]="3.7B" [big]="7.0B"  )

# ---- Data configs ----
# nano/micro: FineWeb-Edu samples (simple, fast)
# ~1090 tokens/sample, so: nano≈55M tok, micro≈545M tok
declare -A CFG_SAMPLES=( [nano]=50000 [micro]=500000 )
# mini+: Multi-corpus total tokens (SmolLM2 recipe)
declare -A CFG_TOKENS=(  [mini]="1500M" [small]="3000M" [goldie]="5000M" [medium]="10000M" [large]="20000M" [big]="40000M" )
# Default corpus per size
declare -A CFG_CORPUS=(  [nano]="fineweb" [micro]="fineweb" [mini]="multi" [small]="multi" [goldie]="multi" [medium]="multi" [large]="multi" [big]="multi" )

# ---- TODO: Future pipeline stages ----
# Mid-training (small+): Separate quality stage with reduced LR after base.
#   SmolLM2 uses curated subset at 10x lower LR. Worth testing for small (336M+).
#   Implementation: extra base_train.py call between base and personality,
#   with --lr-max 0.002 and curated data (filtered CulturaX or SmolTalk).
#
# Multilingual tokenizer tiers (train with: python -m scripts.train_tokenizer --tier N):
#   Tier 1 (goldie, 48K vocab):  EN, RU, FR, DE — Latin + Cyrillic core
#   Tier 2 (medium, 64K vocab):  + ES, PT, UK, TR — extended Latin/Cyrillic
#   Tier 3 (large, 96K vocab):   + AR, HI, ZH, JA, KO — all remaining scripts
#   Each tier inherits previous. Trained on balanced CulturaX samples.
#   γ ⊥ δ proven — personality survives language switch.

# ---- Parse arguments ----
NAME=""
PERSONALITY_FILE=""
PERSONALITY_RATIO=0.20
STEPS_OVERRIDE=""
SAMPLES_OVERRIDE=""
TOKENS_OVERRIDE=""
TAG_SUFFIX=""
CORPUS_OVERRIDE=""
BASE_ONLY=false
SAVE_EVERY=1000
USE_WANDB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)        NAME="$2";              shift 2 ;;
        --personality) PERSONALITY_FILE="$2";   shift 2 ;;
        --ratio)       PERSONALITY_RATIO="$2"; shift 2 ;;
        --steps)       STEPS_OVERRIDE="$2";    shift 2 ;;
        --samples)     SAMPLES_OVERRIDE="$2";  shift 2 ;;
        --tokens)      TOKENS_OVERRIDE="$2";   shift 2 ;;
        --tag)         TAG_SUFFIX="$2";        shift 2 ;;
        --corpus)      CORPUS_OVERRIDE="$2";   shift 2 ;;
        --base-only)   BASE_ONLY=true;         shift ;;
        --save-every)  SAVE_EVERY="$2";        shift 2 ;;
        --wandb)       USE_WANDB=true;         shift ;;
        -h|--help)
            cat <<'HELP'
Usage: bash runs/lambda_train.sh --name <size> [options]

Required:
  --name <size>          nano, micro, mini, small, goldie, medium, large, big

Optional:
  --personality <file>   JSONL for personality training + gamma extraction
  --ratio <0.0-1.0>      Personality data ratio in batches (default: 0.20)
  --steps <N>            Override training steps
  --corpus <type>        fineweb or multi (default: auto by size)
  --samples <N>          Override FineWeb-Edu sample count (fineweb corpus)
  --tokens <N>           Override total tokens, e.g. 500M (multi corpus)
  --tag <suffix>         Custom tag for checkpoints (default: model name)
  --base-only            Skip personality pipeline, train base only
  --save-every <N>       Checkpoint interval (default: 1000)
  --wandb                Enable wandb logging

Sizes:                                                 (~tokens)
  nano     34M   depth=6   ~20 min   1x GPU    FineWeb-Edu 50K      (~55M)
  micro    69M   depth=12  ~40 min   1x GPU    FineWeb-Edu 500K     (~545M)
  mini    150M   depth=16  ~3 hrs    1x GPU    Multi-corpus 500M tokens
  small   336M   depth=24  ~18 hrs   1x GPU    Multi-corpus 1.5B tokens
  goldie  1.1B   depth=20  ~24 hrs   1-2x GPU  Multi-corpus 3B tokens    [4 langs, 48K vocab]
  medium  1.6B   depth=28  ~48 hrs   4x+ GPU   Multi-corpus 5B tokens    [8 langs, 64K vocab]
  large   3.7B   depth=32  ~96 hrs   8x GPU    Multi-corpus 10B tokens   [13 langs, 96K vocab]
  big     7.0B   depth=36  ~200 hrs  8x A100   Multi-corpus 20B tokens   [13 langs, 96K vocab]

Multi-corpus (SmolLM2 recipe, mini+ default):
  FineWeb-Edu 55%  — educational web text
  DCLM 25%         — curated web corpus
  Stack v2 10%     — code (deduped, permissive)
  MegaMath 10%     — mathematical reasoning

Examples:
  bash runs/lambda_train.sh --name mini
  bash runs/lambda_train.sh --name mini --personality wtforacle.jsonl
  bash runs/lambda_train.sh --name small --steps 15000
  bash runs/lambda_train.sh --name nano --base-only
  bash runs/lambda_train.sh --name mini --corpus fineweb --samples 1000000
HELP
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validate ----
if [ -z "$NAME" ]; then
    echo "ERROR: --name is required. Use --help for usage."
    exit 1
fi

if [ -z "${CFG_DEPTH[$NAME]}" ]; then
    echo "ERROR: Unknown model '$NAME'. Available: nano, micro, mini, small, medium, large"
    exit 1
fi

# ---- Resolve config ----
DEPTH=${CFG_DEPTH[$NAME]}
NUM_STEPS=${STEPS_OVERRIDE:-${CFG_STEPS[$NAME]}}
BATCH_SIZE=${CFG_BATCH[$NAME]}
TAG=${TAG_SUFFIX:-$NAME}
PARAMS=${CFG_PARAMS[$NAME]}
CORPUS=${CORPUS_OVERRIDE:-${CFG_CORPUS[$NAME]}}

WANDB_FLAG=""
$USE_WANDB && WANDB_FLAG="--wandb"

# ---- GPU detection ----
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

echo ""
echo "========================================================"
echo "  nanollama — $NAME ($PARAMS, depth=$DEPTH)"
echo "========================================================"
echo "  Steps:       $NUM_STEPS"
echo "  Batch:       $BATCH_SIZE tokens"
if [ "$CORPUS" = "multi" ]; then
    TOTAL_TOKENS=${TOKENS_OVERRIDE:-${CFG_TOKENS[$NAME]}}
    echo "  Corpus:      multi (SmolLM2: FineWeb 55% + DCLM 25% + Code 10% + Math 10%)"
    echo "  Data:        $TOTAL_TOKENS tokens"
else
    NUM_SAMPLES=${SAMPLES_OVERRIDE:-${CFG_SAMPLES[$NAME]:-500000}}
    echo "  Corpus:      FineWeb-Edu"
    echo "  Data:        $NUM_SAMPLES samples"
fi
echo "  GPUs:        $NUM_GPUS x $GPU_NAME"
echo "  Save every:  $SAVE_EVERY steps"
if [ -n "$PERSONALITY_FILE" ] && ! $BASE_ONLY; then
    echo "  Personality: $PERSONALITY_FILE (ratio=$PERSONALITY_RATIO)"
    echo "  Pipeline:    base → personality → gamma → GGUF"
else
    echo "  Pipeline:    base → GGUF"
fi
echo "========================================================"
echo ""

# H100 warning
if echo "$GPU_NAME" | grep -qi "H100"; then
    echo "WARNING: H100 detected — known driver bug (Error 802, Feb 2026)."
    echo "A100 recommended. Ctrl+C to abort."
    sleep 5
fi

# ---- Navigate to repo ----
if [ -f "pyproject.toml" ] && [ -d "nanollama" ]; then
    : # already in repo root
elif [ -d "nanollama" ] && [ -f "nanollama/pyproject.toml" ]; then
    cd nanollama
else
    echo "Cloning nanollama..."
    git clone https://github.com/ariannamethod/nanollama.git
    cd nanollama
fi

# ---- Install deps ----
pip install sentencepiece numpy tqdm datasets filelock 2>/dev/null
pip install . 2>/dev/null || true

# ---- Directories ----
BASE_DIR="$HOME/.cache/nanollama"
TOKENIZER="$BASE_DIR/tokenizer/tokenizer.model"

# ---- Helper: find last checkpoint for a model tag ----
find_last_checkpoint() {
    local tag="$1"
    local ckpt_dir="$BASE_DIR/checkpoints/$tag"
    ls -1 "$ckpt_dir"/checkpoint_step*.pt 2>/dev/null \
        | sed 's/.*checkpoint_step\([0-9]*\)\.pt/\1 &/' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}'
}

# ---- Helper: run training ----
run_training() {
    local model_tag="$1"
    local log_file="$2"
    shift 2
    # remaining args passed through

    echo ""
    echo "  Training: $model_tag"
    echo "  depth=$DEPTH, steps=$NUM_STEPS, batch=$BATCH_SIZE"
    echo ""

    local cmd_args=(
        --depth=$DEPTH
        --run="$model_tag"
        --model-tag="$model_tag"
        --total-batch-size=$BATCH_SIZE
        --num-iterations=$NUM_STEPS
        --save-every=$SAVE_EVERY
        "$@"
    )

    [ -n "$WANDB_FLAG" ] && cmd_args+=($WANDB_FLAG)

    if [ "$NUM_GPUS" -gt 1 ]; then
        OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NUM_GPUS \
            -m scripts.base_train "${cmd_args[@]}" 2>&1 | tee "$log_file"
    else
        python -u -m scripts.base_train "${cmd_args[@]}" 2>&1 | tee "$log_file"
    fi
}

# =====================================================================
# Step 1: Prepare training data
# =====================================================================
if [ "$CORPUS" = "multi" ]; then
    # Multi-corpus (SmolLM2 recipe): FineWeb 55% + DCLM 25% + Stack 10% + MegaMath 10%
    DATA_DIR="$BASE_DIR/data/multi_corpus/merged"
    MULTI_DIR="$BASE_DIR/data/multi_corpus"

    # Check if tokenizer exists (needed before multi_corpus)
    if [ ! -f "$TOKENIZER" ]; then
        echo ">> Step 1a: Training tokenizer on FineWeb-Edu (50K samples)..."
        python -u -m data.prepare_fineweb --num-samples 50000
    fi

    SHARD_COUNT=$(ls "$DATA_DIR"/*.bin 2>/dev/null | wc -l || echo 0)
    if [ "$SHARD_COUNT" -lt 3 ]; then
        echo ">> Step 1: Preparing multi-corpus data ($TOTAL_TOKENS tokens)..."
        echo "   FineWeb-Edu 55% + DCLM 25% + Stack v2 10% + MegaMath 10%"
        python -u -m data.prepare_multi_corpus \
            --total-tokens "$TOTAL_TOKENS" \
            --components fineweb,dclm,stack,megamath
    else
        echo ">> Step 1: Multi-corpus data exists ($SHARD_COUNT shards) — skip"
    fi
else
    # FineWeb-Edu only (nano/micro)
    DATA_DIR="$BASE_DIR/data/fineweb"
    SHARD_COUNT=$(ls "$DATA_DIR"/*.bin 2>/dev/null | wc -l || echo 0)

    if [ "$SHARD_COUNT" -lt 3 ]; then
        echo ">> Step 1: Downloading FineWeb-Edu ($NUM_SAMPLES samples)..."
        python -u -m data.prepare_fineweb --num-samples $NUM_SAMPLES
    else
        echo ">> Step 1: FineWeb-Edu data exists ($SHARD_COUNT shards) — skip"
    fi
fi

# =====================================================================
# Step 2: Prepare personality data (if provided)
# =====================================================================
if [ -n "$PERSONALITY_FILE" ] && ! $BASE_ONLY; then
    # Find personality file
    PERSONALITY_INPUT=""
    for path in "$PERSONALITY_FILE" "data/$PERSONALITY_FILE" "$HOME/$PERSONALITY_FILE"; do
        [ -f "$path" ] && PERSONALITY_INPUT="$path" && break
    done

    if [ -z "$PERSONALITY_INPUT" ]; then
        echo "ERROR: Cannot find personality file: $PERSONALITY_FILE"
        echo "Upload first: scp <file> ubuntu@<ip>:~/"
        exit 1
    fi

    echo ">> Step 2: Preparing personality data ($PERSONALITY_INPUT)..."
    python -u -m data.prepare_personality \
        --input "$PERSONALITY_INPUT" \
        --output-dir "$BASE_DIR/data/personality"
else
    echo ">> Step 2: No personality file — skip"
fi

# =====================================================================
# Step 3: Train BASE model (no personality)
# =====================================================================
echo ""
echo "========================================================"
echo "  Step 3: Training BASE model"
echo "========================================================"

BASE_TAG="${TAG}_base"
TRAIN_ARGS=(--personality-ratio=0.0)

# Point to correct data directory
[ "$CORPUS" = "multi" ] && TRAIN_ARGS+=(--data-dir="$DATA_DIR")

run_training "$BASE_TAG" "train_${TAG}_base.log" "${TRAIN_ARGS[@]}"

# =====================================================================
# Base-only mode: export and exit
# =====================================================================
if $BASE_ONLY || [ -z "$PERSONALITY_FILE" ]; then
    echo ""
    echo "========================================================"
    echo "  Exporting to GGUF"
    echo "========================================================"

    BASE_CKPT=$(find_last_checkpoint "$BASE_TAG")
    if [ -z "$BASE_CKPT" ]; then
        echo "ERROR: No checkpoint found for $BASE_TAG"
        exit 1
    fi

    mkdir -p weights
    python -u -m scripts.export_gguf \
        --checkpoint "$BASE_CKPT" \
        --tokenizer "$TOKENIZER" \
        --output "weights/${TAG}-base-f16.gguf" \
        --dtype f16

    echo ""
    echo "========================================================"
    echo "  DONE — $NAME base ($PARAMS)"
    echo "========================================================"
    echo ""
    echo "  weights/${TAG}-base-f16.gguf"
    echo ""
    echo "  scp ubuntu@\$(hostname -I | awk '{print \$1}'):~/nanollama/weights/${TAG}-* ."
    echo "  ./nanollama --model ${TAG}-base-f16.gguf --interactive"
    exit 0
fi

# =====================================================================
# Step 4: Train PERSONALITY model
# =====================================================================
echo ""
echo "========================================================"
echo "  Step 4: Training PERSONALITY model (ratio=$PERSONALITY_RATIO)"
echo "========================================================"

PERS_TAG="${TAG}_personality"
PERS_ARGS=(
    --personality-dir="$BASE_DIR/data/personality"
    --personality-ratio=$PERSONALITY_RATIO
)
[ "$CORPUS" = "multi" ] && PERS_ARGS+=(--data-dir="$DATA_DIR")

run_training "$PERS_TAG" "train_${TAG}_personality.log" "${PERS_ARGS[@]}"

# =====================================================================
# Step 5: Extract gamma
# =====================================================================
echo ""
echo "========================================================"
echo "  Step 5: Extracting gamma (personality - base)"
echo "========================================================"

BASE_CKPT=$(find_last_checkpoint "$BASE_TAG")
PERS_CKPT=$(find_last_checkpoint "$PERS_TAG")

if [ -z "$BASE_CKPT" ] || [ -z "$PERS_CKPT" ]; then
    echo "ERROR: Cannot find checkpoints"
    echo "  Base:        $BASE_CKPT"
    echo "  Personality: $PERS_CKPT"
    exit 1
fi

mkdir -p weights
GAMMA_FILE="weights/gamma_${TAG}.npz"

python -u -m scripts.extract_gamma \
    --personality_ckpt "$PERS_CKPT" \
    --base_ckpt "$BASE_CKPT" \
    --output "$GAMMA_FILE"

echo "Gamma: $GAMMA_FILE"

# =====================================================================
# Step 6: Export to GGUF
# =====================================================================
echo ""
echo "========================================================"
echo "  Step 6: Exporting to GGUF"
echo "========================================================"

# Personality model (main artifact)
python -u -m scripts.export_gguf \
    --checkpoint "$PERS_CKPT" \
    --tokenizer "$TOKENIZER" \
    --output "weights/${TAG}-f16.gguf" \
    --dtype f16

# Base model (for comparison / gamma injection)
python -u -m scripts.export_gguf \
    --checkpoint "$BASE_CKPT" \
    --tokenizer "$TOKENIZER" \
    --output "weights/${TAG}-base-f16.gguf" \
    --dtype f16

# =====================================================================
# Done
# =====================================================================
echo ""
echo "========================================================"
echo "  DONE — $NAME ($PARAMS) with personality"
echo "========================================================"
echo ""
echo "  Artifacts:"
echo "    weights/${TAG}-f16.gguf          personality model"
echo "    weights/${TAG}-base-f16.gguf     base model"
echo "    weights/gamma_${TAG}.npz         personality vector (gamma)"
echo ""
echo "  Logs:"
echo "    train_${TAG}_base.log"
echo "    train_${TAG}_personality.log"
echo ""
echo "  Download:"
echo "    scp ubuntu@\$(hostname -I | awk '{print \$1}'):~/nanollama/weights/${TAG}* ."
echo ""
echo "  Run:"
echo "    ./nanollama --model ${TAG}-f16.gguf --interactive"
echo "    ./nanollama --model ${TAG}-base-f16.gguf --gamma gamma_${TAG}.npz --interactive"
