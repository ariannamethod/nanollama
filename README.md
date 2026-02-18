# nanollama

**Train your own ChatGPT-like model from scratch for ~$100.**

nanollama is a fork of [Karpathy's nanochat](https://github.com/karpathy/nanochat), rebuilt on **Llama 3 architecture** instead of GPT-2. Same brilliant pipeline (tokenization → pretraining → SFT → RL → eval → web UI), but with a modern, more efficient backbone.

```
                                                       ████  ████
                                                      ░░███ ░░███
     ████████    ██████   ████████    ██████   █████   ░███  ░███   ██████   █████████████    ██████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███  ░███  ░░░░░███ ░░███░░███░░███  ░░░░░███
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███  ░███   ███████  ░███ ░███ ░███   ███████
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███  ░███  ███░░███  ░███ ░███ ░███  ███░░███
     ████ █████░░████████ ████ █████░░██████ ░░██████  █████ █████░░████████ █████░███ █████░░████████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░░ ░░░░░  ░░░░░░░░ ░░░░░ ░░░ ░░░░░  ░░░░░░░░
```

## What is this?

nanochat is brilliant — a complete LLM training pipeline in minimal, hackable code. Train GPT-2 capability for ~$100 on 8×H100. But GPT-2 is 2019 architecture. The world has moved on.

**nanollama keeps everything great about nanochat** (the training stages, eval benchmarks, web UI, CLI, the `--depth` philosophy) **but replaces GPT-2 with Llama 3**.

## Architecture: GPT-2 → Llama 3

Here's what changes under the hood:

| Component | GPT-2 (nanochat) | Llama 3 (nanollama) |
|-----------|-----------------|---------------------|
| Position encoding | Learned embeddings | **RoPE** (Rotary Position Embeddings) |
| Attention | Multi-Head Attention | **GQA** (Grouped Query Attention) |
| FFN activation | GELU | **SwiGLU** |
| Normalization | LayerNorm | **RMSNorm** |
| Norm placement | Post-norm | **Pre-norm** |
| Linear bias | Yes | **No** |
| Embeddings | Tied input/output | **Untied** |
| Dropout | Yes | **No** (Llama 3 doesn't use it) |

### Why these changes matter:

**GQA (Grouped Query Attention)**: Fewer key-value heads than query heads. During inference, the KV cache is smaller → faster generation, lower memory. A 32-head model with 8 KV heads uses 4× less memory for KV cache.

**SwiGLU**: `swish(gate(x)) * up(x)` instead of `gelu(fc(x))`. More expressive, better parameter efficiency. Adds one extra projection but empirically trains better.

**RoPE**: Relative position encoding that extrapolates to longer sequences. No learned position embeddings to store. Works naturally with KV cache.

**Pre-norm**: Normalize before attention and FFN, not after. More stable training at scale.

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/ariannamethod/nanollama.git
cd nanollama

# Install with pip
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

### Train a Model

```bash
# Single GPU - tiny model for testing (~15M params)
python -m scripts.base_train --depth=6

# Multi-GPU - full training run (~350M params, ~3 hours on 8×A100)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=24
```

### Talk to Your Model

```bash
# Web UI (open http://localhost:8000)
python -m scripts.chat_web --model-tag=base

# Command line
python -m scripts.chat_cli --model-tag=base
```

## The `--depth` Parameter

Like nanochat, nanollama is built around a single complexity dial: `--depth` (number of transformer layers). Set this one number and everything else is calculated automatically: width, heads, KV heads, FFN dimension, learning rate, batch size, training horizon.

This makes experimentation trivial. Want to test something? Run `--depth=6` (2 minutes). Looks promising? Scale to `--depth=24` (3 hours). The scaling is calibrated so results transfer.

### Model Series

| Name | Depth | Width | Heads | KV Heads | Params | Notes |
|------|-------|-------|-------|----------|--------|-------|
| nano | 6 | 384 | 6 | 2 | ~15M | Laptop testing |
| micro | 12 | 512 | 8 | 2 | ~50M | Single GPU |
| mini | 16 | 768 | 12 | 4 | ~120M | Single GPU |
| small | 24 | 1024 | 16 | 4 | ~350M | Single A100 |
| medium | 32 | 2048 | 32 | 8 | ~1B | 4×A100 |
| large | 32 | 3200 | 32 | 8 | ~3B | 8×A100 |

The GQA ratio (n_head / n_kv_head) is typically 3-4×, following Llama 3's design.

## Full Pipeline

nanollama preserves nanochat's complete pipeline:

### 1. Tokenization

```bash
# Train a SentencePiece tokenizer
python -m scripts.tok_train --input=data.txt --vocab-size=32000

# Evaluate tokenizer quality
python -m scripts.tok_eval
```

We use SentencePiece BPE (standard for Llama) instead of GPT-4 style tiktoken. You can also plug in Llama 3's original tokenizer.

### 2. Pretraining

```bash
# Standard pretraining on FineWeb
python -m scripts.base_train --depth=24

# With personality injection (unique to nanollama)
python -m scripts.base_train --depth=24 \
    --personality-dir=./personality_data \
    --personality-ratio=0.20
```

**Personality injection**: Mix conversation data directly into pretraining (default 20%). The model learns language and personality simultaneously — no separate fine-tuning stage needed. This is controlled by `--personality-ratio`.

### 3. Supervised Fine-Tuning (SFT)

```bash
python -m scripts.chat_sft --model-tag=base
```

Supports SmolTalk and custom JSON datasets. Uses Llama 3 chat format:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>
```

### 4. Reinforcement Learning (RL)

```bash
python -m scripts.chat_rl --model-tag=sft
```

Same RL pipeline as nanochat, adapted for Llama 3.

### 5. Evaluation

```bash
# Base model: CORE score, bits per byte
python -m scripts.base_eval --model-tag=base

# Chat model: ARC, GSM8K, MMLU, HumanEval, SpellingBee
python -m scripts.chat_eval --model-tag=chat
```

### 6. Inference

```bash
# Web UI
python -m scripts.chat_web

# CLI
python -m scripts.chat_cli
```

The KV cache is optimized for GQA — using fewer KV heads means smaller cache, faster inference.

### 7. Export to GGUF

```bash
# Export for llama.cpp
python -m scripts.export_gguf --model-tag=chat --output=model.gguf
```

Since we use Llama 3 architecture, GGUF export is straightforward. Use llama.cpp for production inference.

## Implementation Details

### llama.py

The model definition is in `nanollama/llama.py`. It's ~500 lines, clean and readable. A student reads this one file and understands Llama 3.

Key classes:
- `LlamaConfig`: Dataclass with all hyperparameters
- `CausalSelfAttention`: GQA implementation with RoPE
- `SwiGLUFFN`: Gate + up + down projections
- `TransformerBlock`: Pre-norm attention + FFN
- `Llama`: Full model with generation

Notable implementation choices:
- QK normalization for training stability
- Logit soft-capping (tanh) to prevent extreme values
- No dropout (following Llama 3)
- RMSNorm without learnable parameters (just functional)

### Optimizer

We use the same Muon + AdamW combination as nanochat:
- **Muon** for matrix parameters (attention and FFN projections)
- **AdamW** for embeddings and scalars

Muon applies orthogonalization to the gradient via Newton-Schulz iteration, which helps training stability and speed.

### Distributed Training

Multi-GPU via PyTorch DDP with:
- Gradient accumulation for large effective batch sizes
- bf16 mixed precision
- torch.compile() for kernel fusion
- Flash Attention via F.scaled_dot_product_attention

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py              # The model (~500 lines)
│   ├── engine.py             # Inference with GQA KV cache
│   ├── tokenizer.py          # SentencePiece wrapper
│   ├── dataloader.py         # Distributed loader + personality mixing
│   ├── dataset.py            # Data utilities
│   ├── optim.py              # Muon + AdamW
│   ├── common.py             # Utilities
│   ├── core_eval.py          # DCLM CORE evaluation
│   ├── checkpoint_manager.py # Save/load
│   └── ui.html               # Chat frontend
├── scripts/
│   ├── base_train.py         # Pretraining
│   ├── base_eval.py          # Base model eval
│   ├── chat_sft.py           # SFT training
│   ├── chat_rl.py            # RL training
│   ├── chat_web.py           # Web UI server
│   ├── chat_cli.py           # CLI interface
│   ├── tok_train.py          # Train tokenizer
│   └── export_gguf.py        # GGUF export
├── runs/
│   ├── speedrun.sh           # Full training run
│   ├── miniseries.sh         # Train all sizes
│   ├── scaling_laws.sh       # Experiments
│   ├── runcpu.sh             # CPU testing
│   └── lambda_setup.sh       # Cloud setup
├── tasks/                    # Eval benchmarks
├── data/                     # Data prep scripts
└── tests/                    # Unit tests
```

## Lambda Cloud Setup

```bash
# One command setup on Lambda A100 instances
bash runs/lambda_setup.sh

# Then run training
bash runs/speedrun.sh
```

**Known issue**: H100 instances have a driver bug (Error 802) as of Feb 2026. Use A100 instead.

## What This Is NOT

- ❌ **NOT a wrapper around Meta's weights** — trains from scratch
- ❌ **NOT a framework** — minimal dependencies, one repo
- ❌ **NOT production inference** — use llama.cpp for that

## Dependencies

Core:
- PyTorch >= 2.0
- SentencePiece
- numpy
- tqdm

Optional:
- wandb (logging)
- FastAPI + uvicorn (web UI)

That's it. No heavyweight frameworks.

## FAQ

**Q: How does this compare to nanochat?**

Same pipeline, different backbone. nanochat uses GPT-2 architecture, nanollama uses Llama 3. The training scripts, eval tasks, and web UI are adapted but conceptually identical.

**Q: Can I load Meta's Llama 3 weights?**

No. This is for training from scratch. The architecture is compatible, but loading pretrained weights isn't the goal.

**Q: Why not just use Llama 3 directly?**

Because training from scratch teaches you how it all works. And you can customize everything — architecture, data, training procedure.

**Q: What's personality injection?**

Mix conversation pairs into pretraining data. Instead of: pretrain → finetune for personality, you do: pretrain with personality mixed in. The model learns language and personality together.

**Q: How long does training take?**

- depth=6 (15M): ~2 minutes on single GPU
- depth=12 (50M): ~15 minutes on single GPU
- depth=24 (350M): ~3 hours on 8×A100

## Contributing

Contributions welcome. Current focus:
1. Speed up time-to-GPT-2 (currently ~3 hours)
2. Better scaling law calibration
3. More eval tasks
4. GGUF export improvements

AI policy: disclosure. Please note any substantial LLM contributions in PRs.

## Acknowledgements

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy — the foundation
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) — optimization techniques
- [PITOMADOM](https://github.com/ariannamethod/pitomadom) — co-creation and inspiration

## Citation

```bibtex
@misc{nanollama,
  title = {nanollama: Llama 3 fork of nanochat},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ariannamethod/nanollama}
}
```

## License

MIT