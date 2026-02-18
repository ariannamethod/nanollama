"""
Model series configurations for nanollama.

Pre-calculated configs for each model size, derived from --depth parameter.

┌────────┬───────┬─────────┬───────┬──────────┬─────────┬────────────────────┐
│  Name  │ Depth │ d_model │ Heads │ KV Heads │ ~Params │       Notes        │
├────────┼───────┼─────────┼───────┼──────────┼─────────┼────────────────────┤
│ nano   │ 6     │ 384     │ 6     │ 2        │ ~15M    │ Smoke test, laptop │
│ micro  │ 12    │ 512     │ 8     │ 2        │ ~50M    │ Single GPU         │
│ mini   │ 16    │ 768     │ 12    │ 4        │ ~120M   │ Single GPU         │
│ small  │ 24    │ 1024    │ 16    │ 4        │ ~350M   │ Single A100        │
│ medium │ 32    │ 2048    │ 32    │ 8        │ ~1B     │ 4×A100             │
│ large  │ 32    │ 3200    │ 32    │ 8        │ ~3B     │ 8×A100             │
└────────┴───────┴─────────┴───────┴──────────┴─────────┴────────────────────┘
"""

from nanollama.llama import LlamaConfig


def get_nano_config() -> LlamaConfig:
    """~15M params, smoke test, laptop"""
    return LlamaConfig(
        n_layer=6,
        n_embd=384,
        n_head=6,
        n_kv_head=2,
        sequence_len=2048,
    )


def get_micro_config() -> LlamaConfig:
    """~50M params, single GPU"""
    return LlamaConfig(
        n_layer=12,
        n_embd=512,
        n_head=8,
        n_kv_head=2,
        sequence_len=2048,
    )


def get_mini_config() -> LlamaConfig:
    """~120M params, single GPU"""
    return LlamaConfig(
        n_layer=16,
        n_embd=768,
        n_head=12,
        n_kv_head=4,
        sequence_len=2048,
    )


def get_small_config() -> LlamaConfig:
    """~350M params, single A100"""
    return LlamaConfig(
        n_layer=24,
        n_embd=1024,
        n_head=16,
        n_kv_head=4,
        sequence_len=2048,
    )


def get_medium_config() -> LlamaConfig:
    """~1B params, 4×A100"""
    return LlamaConfig(
        n_layer=32,
        n_embd=2048,
        n_head=32,
        n_kv_head=8,
        sequence_len=2048,
    )


def get_large_config() -> LlamaConfig:
    """~3B params, 8×A100"""
    return LlamaConfig(
        n_layer=32,
        n_embd=3200,
        n_head=32,
        n_kv_head=8,
        sequence_len=2048,
    )


# Model registry
MODEL_CONFIGS = {
    "nano": get_nano_config,
    "micro": get_micro_config,
    "mini": get_mini_config,
    "small": get_small_config,
    "medium": get_medium_config,
    "large": get_large_config,
}


def get_config(name: str) -> LlamaConfig:
    """Get config by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]()
