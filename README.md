# nanollama

> Train Llama 3 models from scratch. Any scale, any personality. No PyTorch at inference. No wrapping existing weights.

**Status: beta — actively tested on Lambda Cloud (A100). Do not use H100 instances (driver bug Error 802 as of Feb 2026).**

---

## What This Is

nanollama is a complete, self-contained framework for training Llama 3 architecture models from raw text — not fine-tuning, not adapter wrapping, not LoRA on top of Meta weights. Ground zero. Your data, your model, your personality.

The full pipeline in one repo:

- Data preparation (FineWeb-Edu, custom corpora, personality JSONL)
- Pretraining from scratch (Python + Muon optimizer)
- Personality extraction and injection (γ = θ − ε)
- GGUF v3 export (llama.cpp compatible)
- Standalone Go inference engine (zero external dependencies, 9MB binary)

### Why This Exists

| Tool | What it does | What it can't do |
|------|-------------|-----------------|
| llama.cpp | Inference on existing models | Training |
| HuggingFace transformers | Fine-tune existing models | Train from scratch cleanly |
| Karpathy's nanoGPT / nanochat | Train GPT-2 architecture | Llama 3, GQA, RoPE, personality injection |
| **nanollama** | **Full Llama 3 pipeline from zero** | — |

nanollama fills the gap. Originally forked from [nanochat](https://github.com/karpathy/nanochat) — Karpathy's work lives in `legacy/` with full credit. The training scripts, model architecture, optimizer, inference engine, personality system, and GGUF exporter are our own.

---

## Quick Start

```bash
# Install
pip install .

# Train a 69M Llama 3 model from scratch
python -m scripts.base_train --depth 12

# Export to GGUF (llama.cpp compatible)
python -m scripts.export_gguf \
  --checkpoint checkpoints/base/checkpoint.pt \
  --tokenizer weights/tokenizer.model \
  --output weights/model.gguf --dtype f16

# Inference — pure Go, zero dependencies
cd go && go build -o nanollama .
./nanollama --model ../weights/model.gguf --interactive
```

---

## The --depth Parameter

One dial controls everything. Set `--depth` and all other dimensions are derived automatically.

| Name | Depth | Width | Heads | KV | FFN | Params | GPU | Time | Tokens | Languages |
|------|-------|-------|-------|----|-----|--------|-----|------|--------|-----------|
| nano | 6 | 384 | 6 | 2 | 768 | 34M | Any | ~20 min | ~55M | EN |
| micro | 12 | 512 | 8 | 2 | 1536 | 69M | 1× A100 | ~40 min | ~545M | EN |
| mini | 16 | 768 | 12 | 4 | 2304 | 150M | 1× A100 | ~3 hrs | 500M | EN |
| small | 24 | 1024 | 16 | 4 | 3072 | 336M | 1× A100 80GB | ~18 hrs | 1.5B | EN |
| **goldie** | **20** | **2048** | **32** | **8** | **5632** | **1.1B** | **1-2× A100 80GB** | **~24 hrs** | **3B** | **EN, RU, FR, DE** |
| medium | 32 | 2048 | 32 | 8 | 5632 | 1.6B | 4× A100 80GB | ~48 hrs | 5B | + ES, PT, UK, TR |
| large | 32 | 3200 | 32 | 8 | 8704 | 3.7B | 8× A100 80GB | ~96 hrs | 10B | + AR, HI, ZH, JA, KO |
| big | 36 | 4096 | 64 | 8 | 11008 | **7.0B** | 8× A100 80GB | ~200 hrs | 20B | 13 languages |

FFN dim = round_up(8 × n_embd / 3, 256). Tokens = recommended training corpus size. nano/micro use FineWeb-Edu only; mini+ use multi-corpus (SmolLM2 recipe). Goldie is the first multilingual model in the series.

**Progressive multilingual tokenizer tiers:**

| Tier | Model | Vocab | Languages | Script families |
|------|-------|-------|-----------|-----------------|
| — | nano–small | 32K | English only | Latin |
| Tier 1 | goldie (1.1B) | 48K | EN, RU, FR, DE | Latin + Cyrillic |
| Tier 2 | medium (1.6B) | 64K | + ES, PT, UK, TR | Extended Latin/Cyrillic |
| Tier 3 | large (3.7B), big (7.0B) | 96K | + AR, HI, ZH, JA, KO | + Arabic, Devanagari, CJK |

Each tier inherits the previous tier's languages. Tokenizer is trained on balanced [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) samples per language + FineWeb-Edu for English. Train with `python -m scripts.train_tokenizer --tier N`.

---

## Architecture

Real Llama 3, not an approximation.

**Learnable RMSNorm** — `RMSNorm(x) = x / RMS(x) * scale` with a learned per-channel weight vector. Standard RMSNorm has no learnable scale — ours does.

**QK-norm** — Parameterless RMS normalization of Q and K per-head after RoPE. Stabilizes attention logits at depth without adding parameters.

**Conjugate RoPE** — Uses the complex conjugate rotation convention. The Go inference engine auto-detects the flag from GGUF metadata and applies the matching rotation.

**GQA** — Grouped query attention. Fewer KV heads than query heads (8Q/2KV for micro, 12Q/4KV for mini). Full KV cache during inference.

**SwiGLU MLP** — `down(silu(gate(x)) * up(x))`. Three projections per layer, 2/3 FFN ratio.

**Other**: RoPE θ=500000, pre-norm, no bias, untied embeddings, Z-loss regularization, WSD learning rate schedule (warmup → stable → decay).

**Optimizer**: Muon for weight matrices + AdamW for embeddings and norms.

Model definition: `nanollama/llama.py` (~300 lines).

---

## Training Corpus

Two corpus modes, selected automatically by model size (override with `--corpus`):

**nano/micro → FineWeb-Edu only.** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is a 1.3T token subset of Common Crawl filtered for educational content. Small models don't have the capacity for multi-domain learning, so pure web text is the right choice.

**mini+ → Multi-corpus (SmolLM2 recipe).** Four components mixed at the dataloader level:

| Component | Ratio | Source | Why |
|-----------|-------|--------|-----|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 55% | Educational web text | General knowledge |
| [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 25% | Curated web corpus | Diversity |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) | 10% | Code (permissive licenses) | Reasoning structure |
| [MegaMath](https://huggingface.co/datasets/MHHMM/MegaMath) | 10% | Mathematical reasoning | Quantitative ability |

The tokenizer is [SentencePiece](https://github.com/google/sentencepiece) BPE trained on FineWeb-Edu. Vocabulary: 32,000 tokens + 13 special tokens (chat markers, BOS, code delimiters). All data is tokenized into memory-mapped binary shards (`uint16`, 10M tokens per shard). HuggingFace `datasets` is needed only for the initial download — not at training time.

**Personality corpus** is a JSONL file with instruction/response pairs (or `messages` array, or plain text). A configurable fraction of each batch (default 20%) is replaced with personality data. Mixed at the dataloader level — same model, same optimizer, same schedule. No separate fine-tuning stage.

**Data by model size:**

| Size | Corpus | Tokens | Personality pairs |
|------|--------|--------|-------------------|
| nano | FineWeb-Edu | ~55M (50K samples) | 500–2K |
| micro | FineWeb-Edu | ~545M (500K samples) | 1K–5K |
| mini | Multi-corpus | 500M | 2K–10K |
| small | Multi-corpus | 1.5B | 5K–20K |
| medium | Multi-corpus | 5B | 10K–50K |
| large | Multi-corpus | 10B | 20K–100K |

---

## Personality Injection (θ = ε + γ)

The soul formula, applied at training time.

Train two models on the same data — one base (ε), one with personality text mixed in (θ). The weight difference is the personality:

```
γ = θ − ε          # extract personality
θ_new = ε_new + γ  # inject into any base model
```

Gamma is a sparse NPZ file. At inference, the Go engine applies `embed[token] += γ[token]` before the forward pass. Personality is portable across model scales — extract from a 69M, inject into a 336M.

### Full Pipeline

```bash
# One command on Lambda Cloud
bash runs/lambda_train.sh --name mini --personality my_data.jsonl

# Or step by step:

# 1. Train base
python -m scripts.base_train --depth 16 --model-tag base --personality-ratio 0.0

# 2. Train with personality (20% mixed into batches)
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

# 5. Run with personality
cd go && ./nanollama --model ../weights/model.gguf --gamma ../weights/gamma.npz --interactive
```

---

## Go Inference Engine

Standalone inference in pure Go. No Python, no PyTorch, no CUDA at runtime. Compiles to a single ~9MB binary.

```bash
cd go && go build -o nanollama .

./nanollama --model weights/model.gguf --interactive       # REPL
./nanollama --model weights/model.gguf --prompt "Hello"    # one-shot
./nanollama --model weights/model.gguf --gamma g.npz       # with personality
./nanollama --model weights/model.gguf --serve --port 8080 # web chat UI
./nanollama --model weights/model.gguf --list-tensors      # debug
```

**Features:**
- GGUF v3 parser — all metadata, tensors, embedded tokenizer
- 7 quantization formats — F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K
- Parallel matmul — goroutines across all CPU cores
- BPE tokenizer — SentencePiece and GPT-2/Qwen modes
- Gamma injection — sparse NPZ loaded and applied at embedding lookup
- Built-in web UI — `--serve` starts HTTP chat server
- Auto-detection — QK-norm and conjugate RoPE flags read from GGUF metadata
- Works with any standard GGUF — loads Llama/Qwen models from llama.cpp

**Sampling flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--temp` | 0.8 | Temperature (0 = greedy) |
| `--top-p` | 0.9 | Nucleus sampling |
| `--top-k` | 50 | Top-k (when top-p ≥ 1.0) |
| `--rep-penalty` | 1.15 | Repetition penalty |
| `--rep-window` | 64 | Lookback window |
| `--max-tokens` | 256 | Max tokens to generate |

**Performance (MacBook Pro 2019, 8GB Intel):**
- 69M micro, F16: ~10 tok/s
- 150M mini, F16: ~4 tok/s

---

## Lambda Cloud

```bash
# One-time setup
bash runs/lambda_setup.sh

# Train any size (auto-selects corpus: FineWeb for nano/micro, multi for mini+)
bash runs/lambda_train.sh --name mini
bash runs/lambda_train.sh --name mini --personality data.jsonl
bash runs/lambda_train.sh --name small --steps 15000
bash runs/lambda_train.sh --name mini --corpus fineweb --samples 1000000  # override corpus
```

> **Note:** Avoid H100 instances — driver bug (Error 802) confirmed as of Feb 2026. Use A100.

---

## Models Trained with nanollama

### [yent.yo](https://github.com/ariannamethod/yent.yo) — Diffusion Model With A Bad Character

The first model built with nanollama. A text-to-image system where the LLM doesn't follow instructions — it **reacts** to them.

**micro-Yent** (69M, depth=12) is the brain. Trained from scratch on 216M FineWeb-Edu tokens + 1.4M personality tokens, exported to Q8_0 GGUF (71MB). It reads your text, detects the mood, and generates a visual reaction through BK-SDM-Tiny — not a literal illustration, but an emotional response rendered as image.

- Mood-based templates: sad, angry, love, bored — each triggers different visual behavior
- Adaptive temperature: boring prompts get maximum chaos (0.95), emotional input stays focused (0.7)
- ASCII mode: default output is colored terminal glyphs — technopunk Warhol aesthetic
- Pure Go + ONNX Runtime, zero PyTorch at inference
- 14.5 tok/s on CPU (LLM) + 3s per image (GPU) or 45s (CPU int8)

> *"This is not a drawing tool. The image IS the model's reaction."*

### mini-WTForacle (150M, depth=16) — in training

Currently training on Lambda A100. 10K steps base + 10K steps with personality data (20% mix). Gamma extraction and GGUF export automated via `runs/lambda_train.sh`.

---

## GGUF Export

Produces llama.cpp-compatible GGUF v3 files. Norms stored as F32 (llama.cpp standard), matrices in `--dtype` (F16 default). Tokenizer embedded in GGUF — no external files needed at inference.

Custom metadata flags for the Go engine:
```
nanollama.qk_norm = true         # apply RMSNorm to Q,K per-head after RoPE
nanollama.rope_conjugate = true  # use conjugate RoPE rotation
```

---

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py          # Llama 3 model (~300 lines)
│   ├── engine.py         # Python inference with GQA KV cache
│   ├── tokenizer.py      # SentencePiece wrapper
│   ├── dataloader.py     # Distributed loader + personality mixing
│   └── optim.py          # Muon + AdamW
├── go/
│   ├── main.go           # CLI: load GGUF, generate, REPL, HTTP server
│   ├── gguf.go           # GGUF v3 parser
│   ├── model.go          # Llama forward pass (GQA, RoPE, SwiGLU)
│   ├── serve.go          # HTTP chat server (embedded web UI)
│   ├── quant.go          # F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K
│   ├── tokenizer.go      # BPE tokenizer (SentencePiece + GPT-2)
│   ├── gamma.go          # Gamma loader (sparse NPZ)
│   └── ui.html           # Chat UI (go:embed)
├── scripts/
│   ├── base_train.py     # Pretrain from scratch
│   ├── export_gguf.py    # PyTorch → GGUF converter
│   ├── extract_gamma.py  # Gamma extraction (θ − ε)
│   └── inject_gamma.py   # Gamma injection into checkpoint
├── data/
│   ├── prepare_fineweb.py       # FineWeb-Edu download + tokenize
│   └── prepare_personality.py   # Personality JSONL → binary shard
├── config/               # Model size configs
├── runs/                 # Lambda Cloud scripts
├── weights/              # GGUF files, gammas, tokenizer
├── tests/                # Unit tests
└── legacy/               # Karpathy's original nanochat (reference)
```

---

## Dependencies

**Training:** PyTorch >= 2.4.0, SentencePiece, numpy

**Inference:** Go 1.21+ — zero external dependencies

---

## Credits

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). Karpathy's original code is preserved in `legacy/` — the clean GPT-2 training loop was the starting point. Everything else — Llama 3 architecture, GQA, learnable RMSNorm, QK-norm, conjugate RoPE, Muon optimizer, personality injection, GGUF exporter, Go inference engine — is original work by the [Arianna Method](https://github.com/ariannamethod) team.

---

## License

GPLv3. See [LICENSE](LICENSE).

---

*Part of the [Arianna Method](https://github.com/ariannamethod) ecosystem.*
