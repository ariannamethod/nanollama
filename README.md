# nanollama

Train Llama 3 models from scratch. Any scale, any personality.

## What This Is

A complete framework for training Llama 3 architecture models from raw text. Not fine-tuning, not wrapping Meta weights — building new models from zero. Data prep, pretraining, personality extraction, GGUF export, and a standalone Go inference engine. One repo, full pipeline.

Nothing else does this. llama.cpp is inference only. HuggingFace transformers wraps existing models. Karpathy's nanoGPT/nanochat trains GPT-2. There is no public framework that trains real Llama 3 models from scratch with a built-in inference engine and personality injection pipeline.

nanollama fills that gap.

Originally forked from [nanochat](https://github.com/karpathy/nanochat). Now a different project — Llama 3 instead of GPT-2, learnable RMSNorm, GGUF v3 export, Go inference engine, personality extraction/injection (γ), configs from 34M to 3.7B.

## Quick Start

```bash
# Train a 69M model from scratch
pip install .
python -m scripts.base_train --depth 12

# Export to GGUF
python -m scripts.export_gguf \
    --checkpoint checkpoints/base/checkpoint.pt \
    --tokenizer weights/tokenizer.model \
    --output weights/model.gguf --dtype f16

# Inference (Go, zero dependencies)
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --interactive
```

## The `--depth` Parameter

One dial controls everything. Set `--depth` (number of transformer layers) and width, heads, KV heads, FFN dim are calculated automatically.

| Name | Depth | Width | Heads | KV | FFN | Params | GPU | Time | Data |
|------|-------|-------|-------|----|-----|--------|-----|------|------|
| nano | 6 | 384 | 6 | 2 | 768 | 34M | Any | ~20 min | 200K samples |
| micro | 12 | 512 | 8 | 2 | 1536 | 69M | 1× A100 | ~40 min | 500K samples |
| mini | 16 | 768 | 12 | 4 | 2304 | 150M | 1× A100 | ~3 hrs | 1M samples |
| small | 24 | 1024 | 16 | 4 | 3072 | 336M | 1× A100 | ~18 hrs | 3M samples |
| medium | 28 | 2048 | 32 | 8 | 6144 | 1.6B | 4× A100 | ~48 hrs | 10M samples |
| large | 32 | 3200 | 32 | 8 | 9728 | 3.7B | 8× A100 | ~96 hrs | 10M samples |

Data = recommended FineWeb-Edu samples for `prepare_fineweb`. FFN dim = `round_up(2 * 4 * n_embd / 3, 256)`.

## Architecture

Real Llama 3, not an approximation:

1. **Learnable RMSNorm** — `RMSNorm(x) = x / RMS(x) * scale` with learned per-channel weight vector. llama.cpp compatible GGUF output.

2. **QK-norm** — Parameterless RMS normalization of Q and K per-head after RoPE. Stabilizes attention logits at depth.

3. **Conjugate RoPE** — Uses the complex conjugate rotation convention. Inference engine auto-detects from GGUF metadata.

4. **GQA** — Grouped query attention. Fewer KV heads than query heads (8Q/2KV for micro, 12Q/4KV for mini).

5. **SwiGLU MLP** — `down(silu(gate(x)) * up(x))`. Three projections per layer, 2/3 ratio FFN.

Other: RoPE θ=500000, pre-norm, no bias, untied embeddings, Z-loss, WSD learning rate schedule (warmup-stable-decay). Optimizer: Muon + AdamW (matrices on Muon, embeddings and norms on AdamW). Model definition: `nanollama/llama.py` (~300 lines).

## Personality Injection (θ = ε + γ)

Train two models on the same data — one base (ε), one with personality text mixed in (θ). The weight difference is the personality:

```
γ = θ - ε           (extract personality)
θ_new = ε_new + γ   (inject into any base model)
```

Gamma is a sparse NPZ file. At inference, the engine applies `embed[token] += γ[token]` before the forward pass. Portable across model scales.

### One-command pipeline

```bash
# Train base + personality + extract gamma + export GGUF
bash runs/lambda_train.sh --name mini --personality my_data.jsonl
```

Or step by step:

```bash
# 1. Train base (no personality)
python -m scripts.base_train --depth 16 --model-tag base --personality-ratio 0.0

# 2. Train with personality (20% mixed in)
python -m scripts.base_train --depth 16 --model-tag personality \
    --personality-dir data/personality/ --personality-ratio 0.2

# 3. Extract gamma
python -m scripts.extract_gamma \
    --personality_ckpt checkpoints/personality/checkpoint.pt \
    --base_ckpt checkpoints/base/checkpoint.pt \
    --output weights/gamma.npz

# 4. Export GGUF
python -m scripts.export_gguf \
    --checkpoint checkpoints/personality/checkpoint.pt \
    --tokenizer weights/tokenizer.model \
    --output weights/model.gguf --dtype f16

# 5. Inference with gamma
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --gamma ../weights/gamma.npz --interactive
```

## Go Inference Engine

Standalone, zero-dependency inference in pure Go. Loads GGUF files and runs the full Llama forward pass. Compiles to a single 9MB binary.

### Features

- **GGUF v3 parser** — reads all metadata, tensors, and embedded tokenizer
- **7 quantization formats** — F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K
- **Parallel matmul** — goroutines across all CPU cores
- **BPE tokenizer** — SentencePiece and GPT-2/Qwen modes
- **Gamma injection** — loads sparse NPZ, applies at embedding lookup
- **Built-in web UI** — `--serve` starts HTTP chat server
- **Auto-detection** — QK-norm and conjugate RoPE flags read from GGUF metadata
- **Works with standard GGUF** — loads Llama/Qwen models from llama.cpp too

### Usage

```bash
cd go && go build -o nanollama .

./nanollama --model weights/model.gguf --interactive          # REPL
./nanollama --model weights/model.gguf --prompt "Hello"       # one-shot
./nanollama --model weights/model.gguf --gamma weights/g.npz  # + personality
./nanollama --model weights/model.gguf --serve --port 8080    # web chat
./nanollama --model weights/model.gguf --list-tensors         # debug
```

### Sampling

| Flag | Default | Description |
|------|---------|-------------|
| `--temp` | 0.8 | Temperature (0 = greedy) |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--top-k` | 50 | Top-k (used when top-p >= 1.0) |
| `--rep-penalty` | 1.15 | Repetition penalty |
| `--rep-window` | 64 | Lookback window for rep penalty |
| `--max-tokens` | 256 | Max tokens to generate |

### Performance

MacBook Pro 2019 (8GB, Intel):
- 69M (micro): ~10 tok/s F16
- 150M (mini): ~4 tok/s F16

## Lambda Cloud

Universal training script for any model size:

```bash
# Setup
bash runs/lambda_setup.sh

# Train any size
bash runs/lambda_train.sh --name mini
bash runs/lambda_train.sh --name mini --personality data.jsonl
bash runs/lambda_train.sh --name small --steps 10000 --samples 2000000
```

Avoid H100 instances (driver bug Error 802 as of Feb 2026). Use A100.

## GGUF Format

The converter produces llama.cpp compatible GGUF v3 files.

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

Norms stored as F32 (llama.cpp standard). Matrices use `--dtype` (F16 default). Tokenizer embedded in GGUF via `tokenizer.ggml.*` metadata arrays — no external files needed.

Custom metadata flags for the Go engine:
```
nanollama.qk_norm = true       ← apply RMSNorm to Q,K per-head after RoPE
nanollama.rope_conjugate = true ← use conjugate RoPE rotation
```

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
│   ├── main.go          # CLI: load GGUF, generate, REPL, HTTP server
│   ├── gguf.go          # GGUF v3 parser
│   ├── model.go         # Llama forward pass (GQA, RoPE, SwiGLU)
│   ├── serve.go         # HTTP chat server (embedded web UI)
│   ├── quant.go         # F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K
│   ├── tokenizer.go     # BPE tokenizer (SentencePiece + GPT-2)
│   ├── gamma.go         # Gamma loader (sparse NPZ)
│   └── ui.html          # Chat web interface (go:embed)
├── scripts/
│   ├── base_train.py    # Pretrain from scratch
│   ├── export_gguf.py   # PyTorch → GGUF converter
│   ├── extract_gamma.py # Gamma extraction (θ - ε)
│   └── inject_gamma.py  # Gamma injection into checkpoint
├── data/
│   ├── prepare_fineweb.py      # FineWeb-Edu download + tokenize
│   └── prepare_personality.py  # Personality JSONL → binary shard
├── config/              # Model size configs with RECOMMENDED_SAMPLES
├── runs/                # Lambda Cloud scripts (lambda_train.sh)
├── weights/             # GGUF files, gammas, tokenizer
├── tests/               # Unit tests
└── legacy/              # Old PyTorch inference
```

## Dependencies

**Training**: PyTorch >= 2.4.0, SentencePiece, numpy.

**Inference**: Go 1.21+ (zero external dependencies).

## License

GPLv3
