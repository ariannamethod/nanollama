# nanollama

Train your own Llama 3 model from scratch for ~$100. Includes Go inference engine and personality injection (γ).

Fork of [nanochat](https://github.com/karpathy/nanochat) with Llama 3 architecture (RoPE, GQA, SwiGLU, RMSNorm) instead of GPT-2.

## Quick Start

```bash
# Train
pip install -e .
python scripts/base_train.py --depth 12

# Export to GGUF
python scripts/export_gguf.py \
    --checkpoint checkpoints/micro-d12-base/checkpoint_step5000.pt \
    --tokenizer weights/tokenizer.model \
    --output weights/model.gguf --dtype f16

# Inference (Go, zero dependencies)
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --interactive
```

## The `--depth` Parameter

One dial controls everything. Set `--depth` (number of transformer layers) and width, heads, KV heads, FFN dim, learning rate are calculated automatically.

| Name | Depth | Width | Heads | KV Heads | FFN | Params |
|------|-------|-------|-------|----------|-----|--------|
| nano | 6 | 384 | 6 | 2 | 768 | 34M |
| micro | 12 | 512 | 8 | 2 | 1536 | 69M |
| mini | 16 | 768 | 12 | 4 | 2304 | 150M |
| small | 24 | 1024 | 16 | 4 | 3072 | 336M |
| medium | 32 | 2048 | 32 | 8 | 6144 | 1.6B |
| large | 32 | 3200 | 32 | 8 | 9728 | 3.7B |

FFN dim = SwiGLU intermediate size = `round_up(2 * 4 * n_embd / 3, 256)`.

## Architecture

Llama 3 with three training-stabilization extensions:

1. **Parameterless RMSNorm** — `rms_norm(x)` as a plain function without learnable weight vectors. The model learns to use the raw normalized hidden states. This eliminates norm weight storage but requires injecting identity (all-ones) vectors when exporting to GGUF for compatibility.

2. **QK-norm** — After computing Q and K projections and applying RoPE, each head is RMS-normalized independently (without weights). This stabilizes attention logits at large depths by preventing the dot-product magnitude from growing unchecked.

3. **Conjugate RoPE** — The rotary position encoding uses the complex conjugate convention:
   ```
   Standard:  (x0*cos - x1*sin, x0*sin + x1*cos)
   Conjugate: (x0*cos + x1*sin, -x0*sin + x1*cos)
   ```
   Both produce valid position encodings with different attention patterns. The inference engine must match the training convention.

Other details: RoPE θ=500000, GQA (grouped query attention), SwiGLU MLP, pre-norm, no bias on attention, untied embeddings. Model definition: `nanollama/llama.py` (~300 lines).

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
    --model-tag mini-d16-base --num-iterations 5000

# 2. Train personality model (20% personality text mixed in)
python scripts/base_train.py --depth 16 --personality-ratio 0.2 \
    --model-tag mini-d16-personality --personality-dir data/personality/ \
    --num-iterations 5000

# 3. Extract gamma (sparse NPZ)
python scripts/extract_gamma.py \
    --personality_ckpt checkpoints/mini-d16-personality/checkpoint_step5000.pt \
    --base_ckpt checkpoints/mini-d16-base/checkpoint_step5000.pt \
    --output weights/gamma.npz

# 4. Export to GGUF
python scripts/export_gguf.py \
    --checkpoint checkpoints/mini-d16-personality/checkpoint_step5000.pt \
    --tokenizer weights/tokenizer.model \
    --output weights/model.gguf --dtype f16

# 5. Inference with gamma
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --gamma ../weights/gamma.npz --interactive
```

## Go Inference Engine

The `go/` directory contains a standalone, zero-dependency inference engine in pure Go. It loads GGUF files directly and runs the full Llama forward pass.

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

## GGUF Export Details

The converter (`scripts/export_gguf.py`) handles three architectural differences between nanollama and standard GGUF:

### Weight Name Mapping

```
nanollama checkpoint          → GGUF tensor name
─────────────────────────────────────────────────
tok_embeddings.weight         → token_embd.weight
layers.N.attn.c_q.weight      → blk.N.attn_q.weight
layers.N.attn.c_k.weight      → blk.N.attn_k.weight
layers.N.attn.c_v.weight      → blk.N.attn_v.weight
layers.N.attn.c_proj.weight   → blk.N.attn_output.weight
layers.N.ffn.gate_proj.weight → blk.N.ffn_gate.weight
layers.N.ffn.up_proj.weight   → blk.N.ffn_up.weight
layers.N.ffn.down_proj.weight → blk.N.ffn_down.weight
output.weight                 → output.weight
```

### Identity Norm Injection

nanollama uses parameterless RMSNorm, but GGUF consumers expect per-layer norm weight vectors. The converter injects all-ones F32 tensors:

```
blk.N.attn_norm.weight   = ones(n_embd)   ← injected
blk.N.ffn_norm.weight    = ones(n_embd)   ← injected
output_norm.weight        = ones(n_embd)   ← injected
```

### Metadata Flags

Two boolean metadata keys signal the inference engine to handle nanollama's conventions:

```
nanollama.qk_norm = true       ← apply RMSNorm to Q,K per-head after RoPE
nanollama.rope_conjugate = true ← use conjugate RoPE rotation
```

These default to `false` if absent, so standard Llama/Qwen GGUF files work unchanged.

### GGUF Tensor Dimensions

GGUF uses GGML dimension order (innermost first), opposite of PyTorch:
- PyTorch `[768, 512]` → GGUF `[512, 768]`
- The converter reverses dimensions automatically

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

**Note**: Avoid H100 instances (driver bug Error 802 as of Feb 2026). Use A100 80GB.

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py         # Model definition (~300 lines)
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
│   ├── base_train.py    # Pretrain
│   ├── chat_sft.py      # SFT
│   ├── chat_rl.py       # RL
│   ├── export_gguf.py   # PyTorch → GGUF converter
│   ├── extract_gamma.py # Gamma extraction (θ - ε)
│   ├── inject_gamma.py  # Gamma injection into checkpoint
│   └── compare_models.py # Compare two checkpoints
├── config/              # Model configs
├── tasks/               # Eval tasks
├── data/                # Data prep scripts
├── weights/             # GGUF files, gammas, tokenizer
├── tests/               # Smoke tests
└── runs/                # Lambda Cloud scripts
```

## Smoke Test

```bash
python -m tests.smoke_test
```

Trains nano model on random data, verifies loss decreases. ~5 seconds on CPU.

## Dependencies

**Training**: PyTorch >= 2.4.0, SentencePiece, numpy.

**Inference**: Go 1.21+ (zero external dependencies).

## License

GPLv3
