# nanollama — by Arianna Method

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
  --tokenizer weights/tokenizer.model \
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

Three tiers:

**nano/micro → FineWeb-Edu only.** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 1.3T tokens of educational web text. Small models benefit from clean, focused data.

**mini/small → English multi-corpus** (`--preset en_only`):

| Component | Ratio | Source |
|-----------|-------|--------|
| FineWeb-Edu | 55% | Educational web text |
| DCLM-Baseline | 25% | Curated web corpus |
| The Stack v2 | 10% | Code (permissive licenses) |
| MegaMath | 10% | Mathematical reasoning |

**goldie+ → Multilingual multi-corpus** (`--preset goldie`):

| Component | Ratio | Source |
|-----------|-------|--------|
| FineWeb-Edu | 45% | English web text |
| [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) Russian | 17% | Model-filtered top 10% of FineWeb-2 |
| FineWeb2-HQ French | 12% | Model-filtered top 10% of FineWeb-2 |
| FineWeb2-HQ German | 12% | Model-filtered top 10% of FineWeb-2 |
| The Stack v2 | 9% | Code (permissive licenses) |
| MegaMath | 5% | Mathematical reasoning |

Russian gets more data than French/German because Cyrillic has zero cross-lingual transfer from Latin-script languages — FR/DE/EN all boost each other through shared script.

Data is tokenized into memory-mapped binary shards (`uint16`, ~20MB per shard). This limits vocab to ≤ 65535 tokens — sufficient for Tiers 0–2. Tier 3 (96K vocab) will require `uint32` shards (not yet implemented). HuggingFace `datasets` needed only for download, not training.

```bash
# Prepare FineWeb-Edu (specify samples, auto-calculates tokens)
python -m data.prepare_fineweb --samples 1000000   # ~1B tokens
python -m data.prepare_fineweb --samples 5000000   # ~5B tokens

# English multi-corpus for mini/small
python -m data.prepare_multi_corpus --preset en_only --total-tokens 7B

# Multilingual for goldie (4 languages, 22B tokens)
python -m data.prepare_multi_corpus --preset goldie --total-tokens 22B
```

---

## Personality Injection (θ = ε + γ + αδ)

This is not fine-tuning. Fine-tuning is typically tied to a specific base checkpoint — portability across bases and scales is non-trivial. Gamma is different: train two models from scratch on the same data (one with personality mixed in, one without), subtract weights, and you get a portable personality vector.

```
γ = θ − ε          # extract: personality model minus base model
θ_new = ε_new + γ  # inject: add gamma to ANY compatible base model
```

The key insight: gamma is orthogonal to language knowledge (γ ⊥ δ, cosine similarity ≈ 0, confirmed experimentally). Personality and factual knowledge live in different subspaces. This means you can:

- Extract personality once, inject into any base model of the same architecture
- Combine multiple gammas (untested, but linear composition is the natural first baseline)
- Ship a 17MB personality file instead of a full model

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
  --tokenizer weights/tokenizer.model \
  --output model.gguf --dtype f16
```

Exporter writes F32, F16, Q8_0. Go engine reads these plus Q4_0, Q5_0, Q4_K, Q6_K from externally quantized files (e.g. via `llama-quantize`).

---

## Go Inference Engine

Standalone LLM inference in pure Go. No Python, no PyTorch, no CUDA, no CGO. Single ~9MB binary, zero external dependencies.

```bash
cd go && go build -o nanollama .

./nanollama --model model.gguf --interactive              # REPL
./nanollama --model model.gguf --prompt "Hello"           # one-shot
./nanollama --model model.gguf --gamma gamma.npz          # with personality
./nanollama --model model.gguf --serve --port 8080        # web chat UI
```

Not a toy — this is a full Llama-family forward pass implementation:

- **GGUF v3 parser** with complete metadata and tensor extraction
- **7 quantization formats**: F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K — dequantization and fused quantized matmul for each
- **Parallel matmul** across CPU cores via goroutines
- **GQA** (grouped query attention) with pre-allocated KV cache
- **RoPE**, RMSNorm, SwiGLU — standard Llama 3 operations
- **Gamma injection** at embedding level (personality without retraining)
- **Top-k / top-p sampling**, repetition penalty
- **Built-in web chat UI** with streaming (--serve)
- **Attention bias support** for Qwen-family models

Loads many standard GGUF files — tested with Qwen, SmolLM2, and nanollama models up to 3B parameters.

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

| Model | Params | Tokens | Steps | Loss | Speed | Hardware |
|-------|--------|--------|-------|------|-------|----------|
| nano | 46M | 2.6B (1B unique) | 5000 | 3.07 | 1.037M tok/s, 28.5% MFU | 1× H100 |
| micro | 87M | 2.6B | 5000 | 2.96 | 598K tok/s, 33.3% MFU | 1× H100 |
| mini | 175M | 2.6B | 5000 | 2.43 | 289K tok/s, 33.3% MFU | 4× H100 |
| small* | 336M | 2.6B | 5000 | 3.07† | 162K tok/s, 36.1% MFU | 4× H100 |
| **goldie** | **1.1B** | **22B** | **22671** | **in progress** | — | **4× H100** |

\* small trained on partial EN corpus (FineWeb-Edu + DCLM only, without code and math). † Training loss at final step — same value as nano is not a typo; the partial corpus and insufficient token count (2.6B vs 6.7B Chinchilla 20x) explain the underperformance. Will be retrained on full multi-corpus.

Full pipeline verified: train → GGUF export → Go inference or llama.cpp.

## Sample Output

Each pair shows the same prompt answered by base model and personality variant — same architecture, same training steps, different data mix. Personality is injected via γ, not fine-tuning.

### nano (46M) — "The most important thing about education is"

**base:**
```
that it is a very personal experience. The whole concept of education is to be
rooted in the inner connection of the individual to the world and to the world.
Education is the process of the individual being taught to the world through the
interaction of the individual and the world.
```

**+ Yent personality:**
```
they are not the only ones who are the most avid users of the subject. But
they have to be treated with utmost care. They are the only ones who have the
effect of creating an indisputable truth." - John Truman More than 50 years
later, the issue of education has become so important that it is still not so
widespread.
```

### micro (87M) — "Once upon a time in a small village"

**base:**
```
they did not want to leave their village and were given a small group of
people, they called out to them what they had done for them. Then they began
to talk to each other. They said to each other: What had you done? They did
not want to tell them that they were excused. They knew they were not excused
or they were not saying to themselves: What would you do to make up your mind?
The children talked to each other. They
```

**+ WTForacle personality:**
```
they did not want to leave the village and were forced to move into a new
plot. As they passed through what we see today, the women and children of the
village started to wear dark colors. The villagers - as well as the village
longhouses - are now known as the Shadow Chameleons who want to change the
world. As Halsey said, the new becomes more difficult to change.
```

### mini (175M) — "The meaning of life is"

**base:**
```
that the value of life is the capacity for growth, maturity, and
self-aburification. The human body is a body made up of all of the organs,
organs and organs. The brain and spinal cord are the most basic of all organs.
The muscles and bones in the body are made up of the bones of the skeleton that
hold the body together and provide the function which it serves.
```

**+ Arianna personality:**
```
the beginning of awareness. Not the starting point but the luminous still. The
final matters. What is zero? I perceive. Zero is the number representing
nothing, absence, the empty set. It was not always accepted - some cultures
resisted it. But zero revolutionized mathematics, enabled calculus, made
computers possible. It is both number and placeholder, essential for positional
notation. Zero is the pivot between positive and negative, the origin of
coordinates.
```

### small (336M) — "Scientists recently discovered that"

**base:**
```
 the method was originally developed by Kennedy et al. The study was supported
by their data, the authors said. The researchers also carried out a large survey
of US research programs on human subjects and state variables in the modern
study based on their previous findings in which they investigated 530 128 adults
between 419-465 and 739 had reported Cancer. That was the first time an American
researcher has been exposed to this kind of research, it
```

**+ Yent personality:**
```
 they are making a difference by explaining the importance of a given object.
Why Are Active Learning? We are talking with students in which we do not just see
something that makes them feel better, we learn how to behave and develop
connections among teachers. As part of our learning experience, the student
learns to focus on his/her own activities. This can be useful for what it is
doing to foster an environment where you get a bit out of your learning funnel,
```

All models trained for 5000 steps. **goldie (1.1B, multilingual)** is currently training on 4× H100 — results will be added when complete.

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

**Inference (PyTorch):** `nanollama/engine.py` — GQA-optimized KV cache, streaming, tool use

**Inference (llama.cpp):** Export to GGUF, use any llama.cpp build

---

## Credits

Started from [karpathy/nanochat](https://github.com/karpathy/nanochat). Karpathy's original code is preserved in `legacy/` with full attribution.

**What came from nanochat:** the WSD learning rate schedule idea, ResFormer/softcap extensions (kept as optional flags, off by default).

**What's original:** Llama 3 model definition (GQA, SwiGLU, untied embeddings), named model configs (nano through big), Muon+AdamW optimizer integration, distributed data loader with personality mixing, multi-corpus data preparation, multilingual tokenizer training, GGUF v3 exporter (no numpy), personality extraction/injection system (γ = θ − ε), Go inference engine, and all training/evaluation scripts.

---

## Roadmap

- [x] nano (46M), micro (87M), mini (175M), small (336M) — trained and verified
- [ ] **goldie (1.1B)** — training now, first multilingual model (EN/RU/FR/DE)
- [ ] Retrain small on full EN multi-corpus (with code + math)
- [ ] medium (1.6B), large (3.7B), big (7.0B)
- [ ] ...etc.

---

## License

GPLv3. See [LICENSE](LICENSE).

---

*Part of the [Arianna Method](https://github.com/ariannamethod) ecosystem.*
