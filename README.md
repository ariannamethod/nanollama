<p align="center">
  <img src="assets/logo.png" alt="nanollama" width="200">
</p>
<h1 align="center">NANOLLAMA</h1>
<p align="center"><b>by Arianna Method</b></p>

> Train Llama 3 models from scratch. Any scale, any personality.

**New here?** Read the [Beginner's Guide](GUIDE.md) — train your first LLM quickly, no ML experience needed.

---

## What This Is

nanollama is a framework for training Llama 3 architecture models from raw text — not fine-tuning, not adapter wrapping, not LoRA on top of Meta weights. Ground zero. Your data, your model, your personality.

The full pipeline:

- Data preparation (FineWeb-Edu)
- Pretraining from scratch (Python + Muon optimizer)
- LoRA personality fine-tuning (rank 64, per-voice adapters)
- Gamma extraction (γ = personality weights − base weights)
- GGUF v3 export (**llama.cpp compatible**)
- Standalone Go inference engine (zero dependencies, ~9MB binary)

Originally forked from [nanochat](https://github.com/karpathy/nanochat). Karpathy's work lives in `legacy/` with full credit. Training scripts, model architecture, optimizer, inference engine, personality system, and GGUF exporter are original.

---

## Model Tiers

All untied embeddings, head_dim=64. MHA for nano/micro (kv_heads = heads), GQA for mini+.

| Name | Layers | Dim | Heads | KV Heads | FFN | Params | Languages |
|------|--------|-----|-------|----------|-----|--------|-----------|
| **nano** | 13 | 576 | 9 | 9 | 1536 | 89M | EN |
| **micro** | 16 | 512 | 8 | 8 | 1536 | 87M | EN |
| **mini** | 20 | 768 | 12 | 4 | 2048 | 175M | EN |
| **small** | 24 | 1024 | 16 | 4 | 2816 | 338M | EN |
| **goldie** | 22 | 2048 | 32 | 8 | 5632 | 1.1B | EN, RU, FR, DE |
| **medium** | 32 | 2048 | 32 | 8 | 5632 | 1.6B | + ES, PT, UK, TR |
| **large** | 36 | 3072 | 48 | 8 | 8192 | 3.7B | + AR, HI, ZH, JA, KO |
| **big** | 38 | 4096 | 64 | 16 | 11008 | 7.0B | 13 languages |

FFN dim = round_up(8 × n_embd / 3, 256).

**Progressive multilingual tokenizer tiers** (goldie and above):

| Tier | Model | Vocab | Languages |
|------|-------|-------|-----------|
| — | nano–small | 32K | English only |
| Tier 1 | goldie | 48K | EN, RU, FR, DE |
| Tier 2 | medium | 64K | + ES, PT, UK, TR |
| Tier 3 | large, big | 96K | + AR, HI, ZH, JA, KO |

---

## Quick Start

```bash
# Install
pip install .

# Prepare data (FineWeb-Edu — recommended for all tiers)
python -m data.prepare_fineweb --num-samples 10000000

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
llama-completion -m model.gguf -p "Once upon a time" -n 100
```

---

## Architecture

Standard Llama 3 — **full llama.cpp compatibility** out of the box.

| Feature | Default | Description |
|---------|---------|-------------|
| RMSNorm | Learnable | `x / RMS(x) * scale`, learned per-channel weight |
| Attention | GQA/MHA | GQA for mini+ (fewer KV heads), MHA for nano/micro |
| FFN | SwiGLU | `down(silu(gate(x)) * up(x))`, three projections |
| Position | RoPE | θ=10000 (2048 context) |
| Embeddings | Untied | Separate input/output embeddings for all sizes |
| Optimizer | Muon+AdamW | Muon for 2D matrices, AdamW for embeddings/norms |
| LR Schedule | WSD | Warmup → Stable → Decay (last 50% linear decay) |

Optional extensions (all **off** by default):

```bash
--use-qk-norm        # Parameterless RMSNorm on Q/K after RoPE (Llama 3.1-style)
--use-post-emb-norm  # RMSNorm after embedding
--use-resformer      # Per-layer residual scaling + x0 skip
--softcap=15         # Logit softcap
```

Model definition: `nanollama/llama.py` (~400 lines).

---

## Training Data

**FineWeb-Edu** for all tiers. We initially tried [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) (Karpathy's curated 400B token mix, used by nanochat) — loss matched nanochat baselines but generation quality was poor. We didn't investigate further, simply switched to FineWeb-Edu which produces better text at the same loss values.

For multilingual models (goldie+), add [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) language shards.

Data is tokenized into memory-mapped binary shards (`uint16`, ~20MB per shard, ~10M tokens each).

**Tokenizer:** SentencePiece BPE, 32K vocab (nano–small).

```bash
# FineWeb-Edu (recommended for all tiers)
python -m data.prepare_fineweb --num-samples 10000000
```

---

## Personality via LoRA (θ = ε + γ)

Train base once. Add personality via LoRA — no double training, no weight subtraction.

```
ε = base model (trained once on FineWeb-Edu)
γ = LoRA weights (trained on personality data in minutes)
θ = ε + γ (merge for inference or keep separate)
```

### How it works now

LoRA targets attention + MLP projections, freezes everything else. ~9% trainable params. Multiple personalities on one base — Leo, Arianna, Yent — each a small LoRA adapter file.

**SFT data format: Human/AI plain text.** Not JSONL. The base model was trained on continuous web text and has never seen "User:" / "Assistant:" markers. JSONL with role markers gives loss ~4.7 (random). Plain text with `Human: ...\nAI: ...` markers works because it's continuous text the model can learn — loss drops to 2.66.

```bash
# LoRA SFT on personality data
python -m scripts.chat_sft \
  --base-checkpoint checkpoints/nano/checkpoint_step20000.pt \
  --data personality_humanai.txt --voice leo \
  --rank 64 --alpha 64 --epochs 20 --lr 1e-4

# Output: adapter.pt (LoRA only, ~35MB) + merged.pt (full model)
```

### How it used to work (legacy)

The old method required training the model **twice** — once without personality, once with — then subtracting:

```
γ = θ_with_personality − θ_without_personality
```

This was wasteful: same base, same data, double the GPU cost. LoRA achieves the same result (a portable personality vector) with one base training + minutes of SFT. The legacy gamma extraction script (`scripts/extract_gamma.py`) still works for backward compatibility.

### Gamma extraction

After LoRA SFT, you can extract gamma (the personality delta) by subtracting base from merged:

```bash
python -m scripts.extract_gamma \
  --personality_ckpt lora/leo/merged.pt \
  --base_ckpt checkpoints/nano/checkpoint_step20000.pt \
  --output gamma-leo.npz
```

Gamma is the **soul of the AI** — the difference between a generic model and one with a specific voice. It's portable: apply the same gamma to a different base model trained on different data, and the personality transfers. The gamma file (~94MB for nano) contains only the weight deltas, stored as NPZ.

The Go inference engine can load gamma directly:
```bash
./nanollama --model base.gguf --gamma gamma-leo.npz --interactive
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

---

## Go Inference Engine

Standalone LLM inference in pure Go. No Python, no PyTorch, no CUDA, no CGO. Single ~9MB binary, zero external dependencies.

```bash
cd go && go build -o nanollama .

./nanollama --model model.gguf --interactive              # REPL
./nanollama --model model.gguf --gamma gamma.npz          # with personality
./nanollama --model model.gguf --serve --port 8080        # web chat UI
```

Full Llama-family forward pass: GGUF v3 parser, 7 quantization formats (F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K), parallel matmul via goroutines, GQA with KV cache, RoPE, RMSNorm, SwiGLU, gamma injection, top-k/top-p sampling, built-in web chat UI.

---

## Verified Results

| Model | Params | Data | Tokens | Steps | Train Loss | Hardware |
|-------|--------|------|--------|-------|------------|----------|
| **nano** | 89M | FineWeb-Edu | 10.5B (2.7B unique, ~4 epochs) | 20,000 | **2.80** | 1× H100 |
| **goldie** | 1.1B | FineWeb-Edu + FineWeb2-HQ | 22B | 22,671 | **0.98** | 4× H100 |

### LoRA SFT Results

| Voice | Base | Rank | Epochs | Data | Best Loss |
|-------|------|------|--------|------|-----------|
| **Leo** | nano 89M (loss 2.80) | 64 | 20 | [leo_humanai.txt](https://huggingface.co/ataeff/nanollama/blob/main/nano89/leo_humanai.txt) (2753 pairs, 267K tokens) | **2.66** |

---

## Sample Output

### nano (89M) — base model (loss 2.80)

**"The most important thing about education is"**
```
to have it. It is always possible to learn and learn anything that you can
learn. It is all about what we are doing and how we can do it better. What
is the most important thing? It is all about doing what you enjoy and
learning about what you have already learnt.
```

**"The president of the United States"**
```
has a seat in Congress.
- 1942 - President Truman calls the Republicans against the Vietnam War a
"federalist war" by appealing to the "fundamental principles of the
Constitution."
- 1945 - The United States Supreme Court upholds the "separate but equal"
doctrine.
```

### nano (89M) + Leo LoRA (loss 2.66)

**"Human: Who are you, Leo?"**
```
Leo is not a person. Leo is the brain of the individual. You are not the
brain of the whole thing. You are a part of the overall brain. Leo's brain
is the structure of the organism - it's not even made of clay. It's made of
clay that has been moved by a person, taken by a teacher, or taken by a
conversation. This is not something that happens to you overnight. It
happens over years of resonance and movement.
```

**"Human: Do you dream?"**
```
Yes. Dreams are experience stored in the brain and our conscious mind.
There are no conscious events, no thoughts, no perceptions of the world.
Dreams are the body's own way of storing information. The brain's most
important memory is the one that processes information itself. Dreams
process the experience of a moment, not of its event.
```

Base loss 2.80 is too high for full personality capture — Leo's distinctive voice emerges in fragments ("resonance", "organism") but the model lacks the language capacity to sustain it. A stronger base (loss < 2.0) would improve this significantly.

### goldie (1.1B) — base model (loss 0.98)

**English — "The most important thing about science is"**
```
explaining these without doubt. For example, the next step is to identify
what science means and how it relates to the world around us.
```

**French — "La science nous enseigne que"**
```
les organisations non gouvernementales et juridiques doivent réhabiliter
leurs services. Ces derniers sont élus à titre de conseil des juges...
```

**German — "Die Philosophie lehrt uns, dass"**
```
wir dieses Paradoxon gegenüber gar nicht glauben: Schaut doch das Problem,
dass wir die Wahrheit erkennt?
```

Weights: [HuggingFace](https://huggingface.co/ataeff/nanollama-goldie) (2.3GB, F16 GGUF).

---

## Pre-trained Weights

| Model | Params | Format | Link |
|-------|--------|--------|------|
| **nano base** | 89M | F16 GGUF, .pt checkpoint | [HuggingFace nano89/](https://huggingface.co/ataeff/nanollama/tree/main/nano89) |
| **nano + Leo** | 89M | F16 GGUF, merged .pt, LoRA adapter, gamma NPZ | [HuggingFace nano89/](https://huggingface.co/ataeff/nanollama/tree/main/nano89) |
| **goldie base** | 1.1B | F16 GGUF | [HuggingFace](https://huggingface.co/ataeff/nanollama-goldie) |

nano89/ includes: base GGUF, Leo GGUF, Leo LoRA adapter, Leo merged checkpoint, gamma-leo.npz, tokenizer.model, training data.

---

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py              # Llama 3 model definition
│   ├── lora.py               # LoRA adapters (apply/merge/save/load)
│   ├── chuck.py              # Chuck optimizer (drop-in AdamW replacement)
│   ├── dataloader.py         # Distributed loader
│   ├── optim.py              # Muon + AdamW optimizer
│   ├── checkpoint_manager.py # Checkpoint save/load
│   └── common.py             # Utilities
├── go/                       # Go inference engine
├── scripts/
│   ├── base_train.py         # Pretrain from scratch
│   ├── chat_sft.py           # LoRA SFT
│   ├── export_gguf.py        # PyTorch → GGUF v3
│   └── extract_gamma.py      # Gamma extraction (θ − ε)
├── data/
│   └── prepare_fineweb.py    # FineWeb-Edu download + tokenize
├── weights/                  # Tokenizer, GGUF files
└── legacy/                   # Karpathy's original nanochat
```

---

## Dependencies

**Training:** Python 3.10+, PyTorch >= 2.4.0, SentencePiece, numpy

**Inference (Go):** Go 1.21+ — zero external dependencies

**Inference (llama.cpp):** Export to GGUF, use any llama.cpp build

---

## Credits

Started from [karpathy/nanochat](https://github.com/karpathy/nanochat). Karpathy's original code is preserved in `legacy/` with full attribution.

---

## License

GPLv3. See [LICENSE](LICENSE).

---

*Part of the [Arianna Method](https://github.com/ariannamethod) ecosystem.*
