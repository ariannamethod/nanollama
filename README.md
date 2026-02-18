# nanollama

Train your own Llama 3 model from scratch for ~$100.

Fork of [nanochat](https://github.com/karpathy/nanochat) with Llama 3 architecture (RoPE, GQA, SwiGLU, RMSNorm) instead of GPT-2.

## Quick Start

```bash
pip install -e .
python -m data.prepare_tinystories
python -m scripts.base_train --depth=6
python -m scripts.chat_cli --model-tag=base
```

## The `--depth` Parameter

One dial controls everything: `--depth` (number of transformer layers). Set this one number and width, heads, KV heads, FFN dim, learning rate are calculated automatically.

| Name | Depth | Width | Heads | KV Heads | Params |
|------|-------|-------|-------|----------|--------|
| nano | 6 | 384 | 6 | 2 | 34M |
| micro | 12 | 512 | 8 | 2 | 69M |
| mini | 16 | 768 | 12 | 4 | 150M |
| small | 24 | 1024 | 16 | 4 | 336M |
| medium | 32 | 2048 | 32 | 8 | 1.6B |
| large | 32 | 3200 | 32 | 8 | 3.7B |

## Pipeline

```bash
# 1. Tokenizer
python -m scripts.tok_train --input=data.txt --vocab-size=32000

# 2. Pretrain (with optional personality injection)
python -m scripts.base_train --depth=24 --personality-ratio=0.20

# 3. SFT
python -m scripts.chat_sft --model-tag=base

# 4. RL
python -m scripts.chat_rl --model-tag=sft

# 5. Eval
python -m scripts.base_eval --model-tag=base
python -m scripts.chat_eval --model-tag=chat

# 6. Inference
python -m scripts.chat_web  # http://localhost:8000
python -m scripts.chat_cli

# 7. Export
python -m scripts.export_gguf --model-tag=chat --output=model.gguf
```

## Multi-GPU Training

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=24
```

## Lambda Cloud

```bash
bash runs/lambda_setup.sh
bash runs/speedrun.sh
```

**Note**: Avoid H100 instances (driver bug Error 802 as of Feb 2026). Use A100 80GB.

## Architecture

Llama 3: RoPE (θ=500000), GQA, SwiGLU, RMSNorm, pre-norm, no bias, untied embeddings.

The model definition is in `nanollama/llama.py` (~300 lines).

## Project Structure

```
nanollama/
├── nanollama/
│   ├── llama.py      # Model definition
│   ├── engine.py     # Inference with GQA KV cache
│   ├── tokenizer.py  # SentencePiece wrapper
│   ├── dataloader.py # Distributed loader + personality mixing
│   └── optim.py      # Muon + AdamW
├── scripts/
│   ├── base_train.py # Pretrain
│   ├── chat_sft.py   # SFT
│   ├── chat_rl.py    # RL
│   └── export_gguf.py
├── config/           # Model configs
├── tasks/            # Eval tasks
└── data/             # Data prep scripts
```

## Smoke Test

```bash
python -m tests.smoke_test
```

Trains nano model on random data, verifies loss decreases. ~5 seconds on CPU.

## Dependencies

- PyTorch >= 2.4.0
- SentencePiece
- numpy

That's it.

## What This Is NOT

- NOT a wrapper around Meta's weights (trains from scratch)
- NOT a framework
- NOT production inference (use llama.cpp for that)

## License

MIT
