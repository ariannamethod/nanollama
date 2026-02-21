# nanollama

> Train Llama 3 models from scratch. Any scale, any personality.

---

## What This Is

nanollama is a framework for training Llama 3 architecture models from raw text — not fine-tuning, not adapter wrapping, not LoRA on top of Meta weights. Ground zero. Your data, your model, your personality.

The full pipeline:

- Data preparation (FineWeb-Edu, multi-corpus, personality JSONL)
- Pretraining from scratch (Python + Muon optimizer)
- Personality extraction and injection (γ = θ − ε)
- GGUF v3 export (**llama.cpp compatible**)
- Standalone Go inference engine (zero dependencies, ~9MB binary)

### Why This Exists

| Tool | What it does | What it can't do |
|------|-------------|-----------------|
| llama.cpp | Inference on existing models | Training |
| HuggingFace transformers | Fine-tune existing models | Train from scratch cleanly |
| Karpathy's nanoGPT / nanochat | Train GPT-2 architecture | Llama 3, GQA, personality injection |
| **nanollama** | **Full Llama 3 pipeline from zero** | — |

Originally forked from [nanochat](https://github.com/karpathy/nanochat). Karpathy's work lives in `legacy/` with full credit. Training scripts, model architecture, optimizer, inference engine, personality system, and GGUF exporter are original.

---

## Model Series

8 named configs. All untied embeddings, head_dim=64. MHA for nano/micro, GQA for mini+.

| Name | Layers | Dim | Heads | KV Heads | FFN | Params | Chinchilla 20x | Languages |
|------|--------|-----|-------|----------|-----|--------|----------------|-----------|
| **nano** | 12 | 384 | 6 | 6 | 1024 | 46M | 0.9B tok | EN |
| **micro** | 16 | 512 | 8 | 8 | 1536 | 87M | 1.7B tok | EN |
| **mini** | 20 | 768 | 12 | 4 | 2048 | 175M | 3.5B tok | EN |
| **small** | 24 | 1024 | 16 | 4 | 2816 | 336M | 6.7B tok | EN |
| **goldie** | 22 | 2048 | 32 | 8 | 5632 | 1.1B | 22B tok | EN, RU, FR, DE |
| **medium** | 32 | 2048 | 32 | 8 | 5632 | 1.6B | 32B tok | + ES, PT, UK, TR |
| **large** | 36 | 3072 | 48 | 8 | 8192 | 3.7B | 74B tok | + AR, HI, ZH, JA, KO |
| **big** | 38 | 4096 | 64 | 16 | 11008 | 7.0B | 140B tok | 13 languages |

FFN dim = round_up(8 × n_embd / 3, 256). Chinchilla 20x is the minimum recommended training tokens. More is better — for small models, 50-100x is common practice (LLama 3 used ~1875x for 8B).

**Progressive multilingual tokenizer tiers** (goldie and above):

| Tier | Model | Vocab | Languages |
|------|-------|-------|-----------|
| — | nano–small | 32K | English only |
| Tier 1 | goldie | 48K | EN, RU, FR, DE |
| Tier 2 | medium | 64K | + ES, PT, UK, TR |
| Tier 3 | large, big | 96K | + AR, HI, ZH, JA, KO |

Train tokenizer: `python -m scripts.train_tokenizer --tier N`

---

## Quick Start

```bash
# Install
pip install .

# Prepare data (1M samples ≈ 1B tokens, enough for nano/micro)
python -m data.prepare_fineweb --samples 1000000

# Train nano from scratch
python -m scripts.base_train --model-size nano

# Distributed (8x GPU)
torchrun --nproc_per_node=8 -m scripts.base_train --model-size small

# Export to GGUF (llama.cpp compatible)
python -m scripts.export_gguf \
  --checkpoint checkpoints/nano/checkpoint.pt \
  --tokenizer path/to/tokenizer.model \
  --output model.gguf --dtype f16

# Test in llama.cpp
llama-cli -m model.gguf -p "Once upon a time" -n 100
```

---

## Architecture

Standard Llama 3 by default — **full llama.cpp compatibility** out of the box.

| Feature | Default | Description |
|---------|---------|-------------|
| RMSNorm | Learnable | `x / RMS(x) * scale`, learned per-channel weight |
| Attention | GQA/MHA | GQA for mini+ (fewer KV heads), MHA for nano/micro |
| FFN | SwiGLU | `down(silu(gate(x)) * up(x))`, three projections |
| Position | RoPE | θ=10000 (2048 context), interleaved rotation |
| Embeddings | Untied | Separate input/output embeddings for all sizes |
| Optimizer | Muon+AdamW | Muon for 2D matrices, AdamW for embeddings/norms |
| LR Schedule | WSD | Warmup → Stable → Decay (last 50% linear decay) |

### Optional Extensions (nanochat-style, off by default)

These break llama.cpp compatibility. Enable only if you know what you're doing:

```bash
--use-qk-norm        # Parameterless RMSNorm on Q/K after RoPE (Llama 3.1-style)
--use-post-emb-norm  # RMSNorm after embedding
--use-resformer      # Per-layer residual scaling + x0 skip
--softcap=15         # Logit softcap
```

Model definition: `nanollama/llama.py` (~400 lines).

---

## Training Data

Two modes:

**nano/micro → FineWeb-Edu only.** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 1.3T tokens of educational web text. Small models benefit from clean, focused data.

**mini+ → Multi-corpus.** Four components mixed at the dataloader level:

| Component | Ratio | Source |
|-----------|-------|--------|
| FineWeb-Edu | 55% | Educational web text |
| DCLM-Baseline | 25% | Curated web corpus |
| The Stack v2 | 10% | Code (permissive licenses) |
| MegaMath | 10% | Mathematical reasoning |

Data is tokenized into memory-mapped binary shards (`uint16`, ~20MB per shard). HuggingFace `datasets` needed only for download, not training.

```bash
# Prepare FineWeb-Edu (specify samples, auto-calculates tokens)
python -m data.prepare_fineweb --samples 1000000   # ~1B tokens
python -m data.prepare_fineweb --samples 5000000   # ~5B tokens

# Multi-corpus for mini+
python -m data.prepare_multi_corpus --target-tokens 5B
```

---

## Personality Injection (θ = ε + γ + αδ)

Train with personality data mixed into batches. Then extract γ (personality vector) by weight subtraction:

```
γ = θ − ε          # extract personality from trained model
θ_new = ε_new + γ  # inject into any base model
```

Gamma is orthogonal to language knowledge (γ ⊥ δ, confirmed experimentally with cosine similarity ≈ 0).

```bash
# Train base
python -m scripts.base_train --model-size nano --model-tag base

# Train with personality (20% mixed into batches)
python -m scripts.base_train --model-size nano --model-tag personality \
  --personality-dir data/personality/ --personality-ratio 0.2

# Extract gamma
python -m scripts.extract_gamma \
  --personality_ckpt checkpoints/personality/checkpoint.pt \
  --base_ckpt checkpoints/base/checkpoint.pt \
  --output gamma.npz
```

---

## GGUF Export

Produces **llama.cpp-compatible** GGUF v3 files. Norms stored as F32, matrices in `--dtype` (F16 default). SentencePiece tokenizer embedded in the file.

```bash
python -m scripts.export_gguf \
  --checkpoint checkpoints/nano/checkpoint.pt \
  --tokenizer path/to/tokenizer.model \
  --output model.gguf --dtype f16
```

Supported dtypes: F32, F16, Q8_0.

---

## Go Inference Engine

Standalone inference in pure Go. No Python, no PyTorch, no CUDA. Single ~9MB binary.

```bash
cd go && go build -o nanollama .

./nanollama --model model.gguf --interactive              # REPL
./nanollama --model model.gguf --prompt "Hello"           # one-shot
./nanollama --model model.gguf --gamma gamma.npz          # with personality
./nanollama --model model.gguf --serve --port 8080        # web chat UI
```

Features: GGUF v3 parser, 7 quant formats (F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K), parallel matmul, BPE tokenizer, gamma injection, built-in web UI. Loads standard GGUF files (Llama, Qwen, etc).

---

## Lambda Cloud

```bash
# One-time setup
bash runs/lambda_setup.sh

# Train any size
bash runs/lambda_train.sh --name nano
bash runs/lambda_train.sh --name mini --personality data.jsonl
bash runs/lambda_train.sh --name small --steps 15000
```

H100 instances work correctly (as of Feb 2026). 1× H100: ~1M tok/s for nano, ~28% MFU.

---

## Verified Results

| Model | Params | Tokens | Loss | Speed | Hardware |
|-------|--------|--------|------|-------|----------|
| nano | 46M | 1B+ | 3.18 @ 2000 steps | 1M tok/s, 28% MFU | 1× H100 |
| nano GGUF | 46M | — | llama.cpp loads ✓ | 21 tok/s | H100 |

Training in progress — results updated as models complete.

---

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py              # Llama 3 model definition
│   ├── engine.py             # Python inference with GQA KV cache
│   ├── tokenizer.py          # SentencePiece wrapper
│   ├── dataloader.py         # Distributed loader + personality mixing
│   ├── optim.py              # Muon + AdamW optimizer
│   ├── common.py             # Utilities, distributed helpers
│   ├── checkpoint_manager.py # Checkpoint save/load
│   └── core_eval.py          # CORE benchmark evaluation
├── go/                       # Go inference engine
│   ├── main.go, model.go, gguf.go, quant.go, tokenizer.go
│   ├── gamma.go, npy.go      # Gamma/NPZ support
│   └── serve.go, ui.html     # Web chat server
├── scripts/
│   ├── base_train.py         # Pretrain from scratch
│   ├── export_gguf.py        # PyTorch → GGUF v3 converter
│   ├── extract_gamma.py      # Gamma extraction (θ − ε)
│   ├── inject_gamma.py       # Gamma injection
│   ├── train_tokenizer.py    # Multilingual tokenizer training
│   └── quantize_gguf.py      # Post-training quantization
├── data/
│   ├── prepare_fineweb.py    # FineWeb-Edu download + tokenize
│   ├── prepare_multi_corpus.py # Multi-corpus preparation
│   └── prepare_personality.py  # Personality JSONL → binary
├── config/                   # Per-size training configs
├── runs/                     # Lambda Cloud + local training scripts
├── tests/                    # Unit + integration tests
├── weights/                  # GGUF files, gammas, tokenizer
└── legacy/                   # Karpathy's original nanochat (reference)
```

---

## Dependencies

**Training:** Python 3.10+, PyTorch >= 2.4.0, SentencePiece, numpy

**Inference (Go):** Go 1.21+ — zero external dependencies

**Inference (llama.cpp):** Export to GGUF, use any llama.cpp build

---

## Credits

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). Karpathy's original code is preserved in `legacy/`. Training pipeline, Llama 3 architecture, Muon optimizer, personality system, GGUF exporter, and Go inference engine are original work by the [Arianna Method](https://github.com/ariannamethod) team.

---

## License

GPLv3. See [LICENSE](LICENSE).

---

*Part of the [Arianna Method](https://github.com/ariannamethod) ecosystem.*
