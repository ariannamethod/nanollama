# nanollama

Train Llama 3 models from scratch. Any scale, any personality.

A complete from-scratch training framework for the Llama 3 architecture with a zero-dependency Go inference engine. Not a wrapper around Meta weights — trains new models from raw text using the real Llama 3 architecture: RoPE (θ=500K), GQA, SwiGLU, learnable RMSNorm, QK-norm.

Originally forked from [nanochat](https://github.com/karpathy/nanochat), now a different framework: Llama 3 instead of GPT-2, GGUF v3 exporter (llama.cpp compatible) + zero-dependency Go inference engine, personality extraction/injection pipeline (γ), multi-scale configs from 34M to 3.7B.

## Quick Start

```bash
# Train a 69M model from scratch
pip install -e .
python scripts/base_train.py --depth 12

# Export to GGUF
python scripts/export_gguf.py \
    --checkpoint checkpoints/micro-d12/checkpoint.pt \
    --tokenizer weights/tokenizer.model \
    --output weights/model.gguf --dtype f16

# Inference (Go, zero dependencies)
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --interactive
```

## The `--depth` Parameter

One dial controls everything. Set `--depth` (number of transformer layers) and width, heads, KV heads, FFN dim are calculated automatically.

| Name | Depth | Width | Heads | KV Heads | FFN | Params | GPU | Time |
|------|-------|-------|-------|----------|-----|--------|-----|------|
| nano | 6 | 384 | 6 | 2 | 768 | 34M | 1× A100 | ~20 min |
| micro | 12 | 512 | 8 | 2 | 1536 | 69M | 1× A100 | ~40 min |
| mini | 16 | 768 | 12 | 4 | 2304 | 150M | 1× A100 | ~1.5 hrs |
| small | 24 | 1024 | 16 | 4 | 3072 | 336M | 8× A100 | ~3 hrs |
| medium | 28 | 2048 | 32 | 8 | 6144 | 1.6B | 8× A100 | ~12 hrs |
| large | 32 | 3200 | 32 | 8 | 9728 | 3.7B | 8× A100 | ~24 hrs |

FFN dim = SwiGLU intermediate size = `round_up(2 * 4 * n_embd / 3, 256)`.

## Architecture

Real Llama 3, not an approximation. Every component matches the published architecture:

1. **RMSNorm** — `RMSNorm(x) = x / RMS(x) * scale` where `scale` is a learned per-channel weight vector. Standard Llama 3, llama.cpp compatible GGUF output.

2. **QK-norm** — After computing Q and K projections and applying RoPE, each head is RMS-normalized independently (parameterless). Stabilizes attention logits at large depths.

3. **Conjugate RoPE** — The rotary position encoding uses the complex conjugate convention:
   ```
   Standard:  (x0*cos - x1*sin, x0*sin + x1*cos)
   Conjugate: (x0*cos + x1*sin, -x0*sin + x1*cos)
   ```
   Both produce valid position encodings with different attention patterns. The inference engine must match the training convention.

4. **GQA** — Grouped query attention. Fewer KV heads than query heads (e.g. 8Q/2KV for micro, 12Q/4KV for mini). Standard for all Llama 3 models.

5. **SwiGLU MLP** — `down(silu(gate(x)) * up(x))`. Three projections per layer, 2/3 ratio FFN.

Other details: RoPE θ=500000, pre-norm, no bias, untied embeddings, Z-loss for logit stabilization. Optimizer: Muon + AdamW (matrices on Muon, embeddings and norms on AdamW). Model definition: `nanollama/llama.py` (~300 lines).

## Personality Injection (θ = ε + γ)

Train two models on the same data — one base (ε), one with personality text mixed in (θ). Extract gamma:

```
γ = θ - ε    (weight difference = personality essence)
```

Then inject gamma into any base model:

```
θ_new = ε_new + γ    (new base model inherits personality)
```

In practice, gamma lives in the embedding layer — it's a sparse diff of which token embeddings shifted during personality training. At inference time, the engine does `embed[token] += γ[token]` before the forward pass.

### Full Pipeline

```bash
# 1. Train base model (no personality)
python scripts/base_train.py --depth 16 --personality-ratio 0.0 \
    --model-tag mini-base --num-iterations 5000

# 2. Train personality model (20% personality text mixed in)
python scripts/base_train.py --depth 16 --personality-ratio 0.2 \
    --model-tag mini-personality --personality-dir data/personality/ \
    --num-iterations 5000

# 3. Extract gamma (sparse NPZ)
python scripts/extract_gamma.py \
    --personality_ckpt checkpoints/mini-personality/checkpoint.pt \
    --base_ckpt checkpoints/mini-base/checkpoint.pt \
    --output weights/gamma.npz

# 4. Export to GGUF
python scripts/export_gguf.py \
    --checkpoint checkpoints/mini-personality/checkpoint.pt \
    --tokenizer weights/tokenizer.model \
    --output weights/model.gguf --dtype f16

# 5. Inference with gamma
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --gamma ../weights/gamma.npz --interactive
```

## Go Inference Engine

The `go/` directory contains a standalone, zero-dependency inference engine in pure Go. Loads GGUF files directly and runs the full Llama forward pass.

### Features

- **GGUF v3 parser** — reads all metadata, tensors, and embedded tokenizer
- **Quantization support** — F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K
- **Parallel matmul** — goroutines across all CPU cores
- **BPE tokenizer** — SentencePiece and GPT-2/Qwen modes
- **Gamma injection** — loads sparse NPZ, applies at embedding lookup
- **nanollama flags** — QK-norm and conjugate RoPE auto-detected from GGUF metadata
- **Streaming output** — tokens printed as they're generated
- **3.5 MB binary** — compiles to a single static binary, no shared libs

### Build & Run

```bash
cd go
go build -o nanollama .

# Generate text
./nanollama --model ../weights/model.gguf --prompt "Once upon a time" --temp 0.7

# Interactive mode
./nanollama --model ../weights/model.gguf --interactive

# With personality
./nanollama --model ../weights/model.gguf --gamma ../weights/gamma.npz --interactive

# Debug: list tensors
./nanollama --model ../weights/model.gguf --list-tensors

# All flags
./nanollama --help
```

### Sampling Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--temp` | 0.8 | Temperature (0 = greedy) |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--top-k` | 50 | Top-k (used when top-p >= 1.0) |
| `--rep-penalty` | 1.15 | Repetition penalty |
| `--rep-window` | 64 | Lookback window for rep penalty |
| `--max-tokens` | 256 | Max tokens to generate |

### Performance

On MacBook Pro 2019 (8GB, Intel):
- 69M model (micro, depth=12): **~10 tok/s** with F16 weights
- 150M model (mini, depth=16): **~4 tok/s** with F16 weights

## GGUF Format

The converter (`scripts/export_gguf.py`) produces llama.cpp compatible GGUF v3 files.

### Weight Name Mapping

```
nanollama checkpoint              → GGUF tensor name
──────────────────────────────────────────────────────
tok_embeddings.weight             → token_embd.weight
layers.N.attn_norm.weight         → blk.N.attn_norm.weight    (F32)
layers.N.attn.c_q.weight          → blk.N.attn_q.weight
layers.N.attn.c_k.weight          → blk.N.attn_k.weight
layers.N.attn.c_v.weight          → blk.N.attn_v.weight
layers.N.attn.c_proj.weight       → blk.N.attn_output.weight
layers.N.ffn_norm.weight          → blk.N.ffn_norm.weight     (F32)
layers.N.ffn.gate_proj.weight     → blk.N.ffn_gate.weight
layers.N.ffn.up_proj.weight       → blk.N.ffn_up.weight
layers.N.ffn.down_proj.weight     → blk.N.ffn_down.weight
norm.weight                       → output_norm.weight         (F32)
output.weight                     → output.weight
```

Norm weights are always stored as F32 (llama.cpp standard). Matrix weights use the specified `--dtype` (F16 by default).

### Metadata Flags

Two boolean metadata keys signal the inference engine to handle nanollama-specific conventions:

```
nanollama.qk_norm = true       ← apply RMSNorm to Q,K per-head after RoPE
nanollama.rope_conjugate = true ← use conjugate RoPE rotation
```

These default to `false` if absent, so standard Llama/Qwen GGUF files work unchanged in the Go engine.

### Embedded Tokenizer

The SentencePiece tokenizer is embedded directly in the GGUF file via metadata arrays:
- `tokenizer.ggml.tokens` — string array of all vocab pieces
- `tokenizer.ggml.scores` — float32 array of BPE scores
- `tokenizer.ggml.token_type` — int32 array (1=normal, 2=unknown, 3=control, 6=byte)

This makes the GGUF file fully self-contained — no external tokenizer file needed.

## Multi-GPU Training

```bash
torchrun --nproc_per_node=8 scripts/base_train.py --depth 24
```

## Lambda Cloud

```bash
bash runs/lambda_setup.sh
bash runs/speedrun.sh
```

**Note**: Avoid H100 instances (driver bug Error 802 as of Feb 2026). Use A100.

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py         # Llama 3 model (~300 lines)
│   ├── engine.py        # Python inference with GQA KV cache
│   ├── tokenizer.py     # SentencePiece wrapper
│   ├── dataloader.py    # Distributed loader + personality mixing
│   └── optim.py         # Muon + AdamW
├── go/
│   ├── main.go          # CLI: load GGUF, generate, REPL
│   ├── gguf.go          # GGUF v3 parser (all types)
│   ├── model.go         # Llama forward pass (GQA, RoPE, SwiGLU)
│   ├── quant.go         # F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K matmul
│   ├── tokenizer.go     # BPE tokenizer (SentencePiece + GPT-2)
│   ├── gamma.go         # Gamma essence loader (sparse NPZ)
│   └── npy.go           # NPY file reading utilities
├── scripts/
│   ├── base_train.py    # Pretrain from scratch
│   ├── chat_sft.py      # Supervised fine-tuning
│   ├── chat_rl.py       # RL fine-tuning
│   ├── chat_web.py      # Web UI (FastAPI)
│   ├── export_gguf.py   # PyTorch → GGUF converter (llama.cpp compatible)
│   ├── extract_gamma.py # Gamma extraction (θ - ε)
│   └── inject_gamma.py  # Gamma injection into checkpoint
├── config/              # Model size configs (nano → large)
├── tasks/               # Eval tasks
├── data/                # Data prep scripts
├── weights/             # GGUF files, gammas, tokenizer
├── tests/               # Unit tests
├── legacy/              # Old PyTorch inference (replaced by Go engine)
└── runs/                # Lambda Cloud scripts
```

## Tests

```bash
python -m pytest tests/ -v
```

## Dependencies

**Training**: PyTorch >= 2.4.0, SentencePiece, numpy.

**Inference**: Go 1.21+ (zero external dependencies).

## License

GPLv3
