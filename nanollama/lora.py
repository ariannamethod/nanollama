"""
LoRA — Low-Rank Adaptation for nanollama.

Inspired by Yent (github.com/ariannamethod/yent):
  LoRA rank 64, delta extraction, per-voice adapters.

Usage:
    from nanollama.lora import apply_lora, merge_lora, save_lora, load_lora

    apply_lora(model, rank=64, alpha=64)    # inject LoRA into attention + FFN
    # ... train only lora params ...
    save_lora(model, "voice.lora.pt")       # save adapter (~2-5% of base)
    merge_lora(model)                        # merge into base for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# Default targets: all projections in CausalSelfAttention + SwiGLUFFN
DEFAULT_TARGETS = ["c_q", "c_k", "c_v", "c_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    """Linear layer with low-rank adapter.

    forward(x) = base(x) + (x @ A^T @ B^T) * scaling

    A is initialized with Kaiming uniform, B with zeros,
    so the adapter starts as identity (zero contribution).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: A (rank x in), B (out x rank) — same device as base
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A)

        # Freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA into base weights and return plain Linear."""
        with torch.no_grad():
            # W_merged = W_base + B @ A * scaling
            self.base.weight.add_(self.lora_B @ self.lora_A * self.scaling)
        return self.base

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def apply_lora(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 64.0,
    target_modules: Optional[List[str]] = None,
) -> int:
    """Inject LoRA adapters into model. Returns count of adapted layers.

    Freezes all base parameters. Only LoRA A/B matrices are trainable.
    """
    targets = target_modules or DEFAULT_TARGETS
    count = 0

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Replace target Linear modules with LoRALinear
    for name, module in model.named_modules():
        for attr in targets:
            if hasattr(module, attr):
                base_linear = getattr(module, attr)
                if isinstance(base_linear, nn.Linear):
                    lora_linear = LoRALinear(base_linear, rank, alpha)
                    setattr(module, attr, lora_linear)
                    count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {count} layers, rank={rank}, alpha={alpha}")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return count


def merge_lora(model: nn.Module) -> int:
    """Merge all LoRA adapters back into base weights. Returns count merged."""
    count = 0
    for name, module in model.named_modules():
        for attr in DEFAULT_TARGETS + list(vars(module).keys()):
            try:
                child = getattr(module, attr)
            except (AttributeError, RuntimeError):
                continue
            if isinstance(child, LoRALinear):
                merged = child.merge()
                setattr(module, attr, merged)
                count += 1
    print(f"LoRA merged: {count} layers")
    return count


def save_lora(model: nn.Module, path: str) -> int:
    """Save only LoRA parameters to file. Returns count of saved tensors."""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.cpu()
    torch.save(lora_state, path)
    print(f"LoRA saved: {len(lora_state)} tensors to {path}")
    return len(lora_state)


def load_lora(model: nn.Module, path: str) -> int:
    """Load LoRA parameters from file. Model must have LoRA already applied."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    loaded = 0
    model_state = dict(model.named_parameters())
    for name, tensor in lora_state.items():
        if name in model_state:
            model_state[name].data.copy_(tensor)
            loaded += 1
        else:
            print(f"WARNING: {name} not found in model")
    print(f"LoRA loaded: {loaded}/{len(lora_state)} tensors from {path}")
    return loaded


def lora_params(model: nn.Module, lr: float = 1e-4) -> List[Dict]:
    """Get parameter groups for LoRA training. Only trainable (LoRA) params."""
    params = [p for p in model.parameters() if p.requires_grad]
    return [{"params": params, "lr": lr}]
