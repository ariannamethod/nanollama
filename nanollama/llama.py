"""
Llama 3 model implementation for nanollama.

A clean, minimal implementation of the Llama 3 architecture designed for
training from scratch. This replaces GPT-2 architecture from nanochat with:

┌───────────────────────────────┬───────────────────────────────────────┐
│       GPT-2 (nanochat)        │          Llama 3 (nanollama)          │
├───────────────────────────────┼───────────────────────────────────────┤
│ Learned positional embeddings │ RoPE (Rotary Position Embeddings)     │
│ Multi-Head Attention          │ GQA (Grouped Query Attention)         │
│ GELU activation               │ SwiGLU (gate + up + down projections) │
│ LayerNorm                     │ RMSNorm                               │
│ Post-norm                     │ Pre-norm (norm before attn and FFN)   │
│ Bias in linear layers         │ No bias                               │
│ Tied embeddings               │ Untied input/output embeddings        │
└───────────────────────────────┴───────────────────────────────────────┘

Notable features:
- RoPE (Rotary Position Embeddings)
- GQA (Grouped Query Attention) for efficient inference
- SwiGLU activation in FFN
- RMSNorm (no learnable params)
- Pre-norm architecture
- No bias in linear layers
- Untied input/output embeddings
- No dropout (Llama 3 style)
- norm_eps = 1e-5

AI policy: disclosure - This code was written with substantial AI assistance.
"""

from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanollama.common import get_dist_info, print0
from nanollama.optim import MuonAdamW, DistMuonAdamW

# -----------------------------------------------------------------------------
# Model Configuration

@dataclass
class LlamaConfig:
    """Configuration for Llama 3 model."""
    sequence_len: int = 2048
    vocab_size: int = 32000  # Default Llama vocab size
    n_layer: int = 12
    n_head: int = 12  # Number of query heads
    n_kv_head: int = 4  # Number of key/value heads (GQA)
    n_embd: int = 768
    norm_eps: float = 1e-5
    multiple_of: int = 256  # FFN hidden dim rounded to this
    # Sliding window attention pattern (from nanochat)
    window_pattern: str = "L"  # L=long (full context), S=short (half context)


# -----------------------------------------------------------------------------
# Model Series (like nanochat's miniseries but for Llama 3)
#
# ┌────────┬───────┬─────────┬───────┬──────────┬─────────┬────────────────────┐
# │  Name  │ Depth │ d_model │ Heads │ KV Heads │ ~Params │       Notes        │
# ├────────┼───────┼─────────┼───────┼──────────┼─────────┼────────────────────┤
# │ nano   │ 6     │ 384     │ 6     │ 2        │ ~15M    │ Smoke test, laptop │
# │ micro  │ 12    │ 512     │ 8     │ 2        │ ~50M    │ Single GPU         │
# │ mini   │ 16    │ 768     │ 12    │ 4        │ ~120M   │ Single GPU         │
# │ small  │ 24    │ 1024    │ 16    │ 4        │ ~350M   │ Single A100        │
# │ medium │ 32    │ 2048    │ 32    │ 8        │ ~1B     │ 4×A100             │
# │ large  │ 32    │ 3200    │ 32    │ 8        │ ~3B     │ 8×A100             │
# └────────┴───────┴─────────┴───────┴──────────┴─────────┴────────────────────┘

def get_config_for_depth(depth: int) -> LlamaConfig:
    """
    Get optimal Llama 3 config for a given depth.
    This is the nanollama equivalent of nanochat's --depth parameter.
    
    The scaling is calibrated for Llama 3 architecture which is more
    parameter-efficient due to GQA and SwiGLU.
    """
    # Width scaling: roughly sqrt scaling with depth
    if depth <= 6:
        n_embd = 384
        n_head = 6
        n_kv_head = 2
    elif depth <= 12:
        n_embd = 512
        n_head = 8
        n_kv_head = 2
    elif depth <= 16:
        n_embd = 768
        n_head = 12
        n_kv_head = 4
    elif depth <= 24:
        n_embd = 1024
        n_head = 16
        n_kv_head = 4
    elif depth <= 32:
        # Scale width based on depth in this range
        if depth <= 28:
            n_embd = 2048
            n_head = 32
            n_kv_head = 8
        else:
            n_embd = 3200
            n_head = 32
            n_kv_head = 8
    else:
        # Beyond depth 32, scale width more aggressively
        n_embd = min(4096, 256 * (depth // 4))
        n_head = min(64, n_embd // 64)
        n_kv_head = max(4, n_head // 4)
    
    return LlamaConfig(
        n_layer=depth,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
    )


# -----------------------------------------------------------------------------
# RMSNorm (no learnable parameters, following Llama 3)

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),), eps=eps)


# -----------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)

def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency tensor for rotary embeddings.
    
    Returns cos and sin tensors of shape (1, seq_len, 1, dim//2) for broadcasting.
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    # Compute position indices
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    # Outer product to get frequencies for each position
    freqs = torch.outer(t, inv_freq)
    # Get cos and sin
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    # Add batch and head dims for broadcasting: (1, T, 1, D/2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    x: (B, T, H, D) - query or key tensor
    cos, sin: (1, T, 1, D/2) - precomputed frequencies
    """
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Apply rotation
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


# -----------------------------------------------------------------------------
# Grouped Query Attention (GQA)

class CausalSelfAttention(nn.Module):
    """
    Grouped Query Attention (GQA) module.
    
    GQA uses fewer key-value heads than query heads, making KV-cache
    more memory efficient during inference.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.norm_eps = config.norm_eps
        
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        
        # Query, Key, Value projections (no bias for Llama 3)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Number of query heads per KV head (for GQA expansion)
        self.n_rep = self.n_head // self.n_kv_head

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        window_size: Tuple[int, int],
        kv_cache=None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        
        # Project to Q, K, V
        # Shape: (B, T, H, D) - Flash Attention's native layout
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Apply QK norm (helps with training stability)
        q = rms_norm(q, self.norm_eps)
        k = rms_norm(k, self.norm_eps)
        
        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            # Use Flash Attention with KV cache
            y = self._flash_attn_with_kvcache(q, k, v, k_cache, v_cache, kv_cache, window_size)
            # Advance position after last layer
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        else:
            # Training: expand KV heads for GQA and use standard attention
            y = self._attention(q, k, v, window_size)
        
        # Reshape and project output
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Standard causal attention with GQA expansion."""
        B, T, _, _ = q.shape
        
        # Expand KV heads to match query heads for GQA
        if self.n_rep > 1:
            # (B, T, n_kv_head, D) -> (B, T, n_head, D)
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use scaled_dot_product_attention with causal mask
        # This uses Flash Attention when available
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,  # No dropout in Llama 3
            is_causal=True,
        )
        
        # Transpose back: (B, T, H, D)
        y = y.transpose(1, 2)
        return y
    
    def _flash_attn_with_kvcache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_cache,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Flash attention with KV cache for inference."""
        # For now, use the same approach as training but with cached KV
        # This can be optimized with flash_attn library
        B, T = q.shape[:2]
        pos = kv_cache.get_pos()
        
        # Store new K, V in cache
        k_cache[:, pos:pos+T, :, :] = k
        v_cache[:, pos:pos+T, :, :] = v
        
        # Get full K, V from cache
        k_full = k_cache[:, :pos+T, :, :]
        v_full = v_cache[:, :pos+T, :, :]
        
        # Expand KV heads
        if self.n_rep > 1:
            k_full = k_full.repeat_interleave(self.n_rep, dim=2)
            v_full = v_full.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, H, T, D)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)
        
        # Causal attention
        y = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        
        y = y.transpose(1, 2)
        return y


# -----------------------------------------------------------------------------
# SwiGLU FFN (Llama 3 style)

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in Llama 3.
    
    FFN(x) = down_proj(swish(gate_proj(x)) * up_proj(x))
    
    Hidden dim = int(8/3 * d_model) rounded to nearest multiple_of (256)
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        n_embd = config.n_embd
        
        # Compute hidden dimension (Llama 3 style: 8/3 * n_embd, rounded)
        hidden_dim = int(2 * (4 * n_embd) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate(x)) * up(x), then down projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Transformer Block

class TransformerBlock(nn.Module):
    """
    Llama 3 transformer block with pre-norm architecture.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_eps = config.norm_eps
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ffn = SwiGLUFFN(config)
    
    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        window_size: Tuple[int, int],
        kv_cache=None,
    ) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attn(rms_norm(x, self.norm_eps), cos_sin, window_size, kv_cache)
        # Pre-norm + FFN + residual
        x = x + self.ffn(rms_norm(x, self.norm_eps))
        return x


# -----------------------------------------------------------------------------
# Main Llama Model

class Llama(nn.Module):
    """
    Llama 3 model for training from scratch.
    
    This is NOT a wrapper around Meta's Llama weights - it trains FROM SCRATCH.
    """
    
    def __init__(self, config: LlamaConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        
        # Compute window sizes for sliding window attention
        self.window_sizes = self._compute_window_sizes(config)
        
        # Pad vocab for efficiency (tensor cores, DDP)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        
        # Token embeddings (input)
        self.tok_embeddings = nn.Embedding(padded_vocab_size, config.n_embd)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.n_layer)
        ])
        
        # Output projection (untied from embeddings)
        self.output = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        
        # Precompute rotary embeddings (10X over-compute for flexibility)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    @torch.no_grad()
    def init_weights(self):
        """
        Initialize model weights following Llama conventions.
        """
        n_embd = self.config.n_embd
        n_layer = self.config.n_layer
        
        # Embedding: normal with std=1.0
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)
        
        # Output projection: small init
        nn.init.normal_(self.output.weight, mean=0.0, std=0.001)
        
        # Transformer blocks: uniform init
        s = (3 ** 0.5) * (n_embd ** -0.5)  # Same std as normal
        for layer in self.layers:
            # Attention projections
            nn.init.uniform_(layer.attn.c_q.weight, -s, s)
            nn.init.uniform_(layer.attn.c_k.weight, -s, s)
            nn.init.uniform_(layer.attn.c_v.weight, -s, s)
            nn.init.zeros_(layer.attn.c_proj.weight)
            
            # FFN projections
            nn.init.uniform_(layer.ffn.gate_proj.weight, -s, s)
            nn.init.uniform_(layer.ffn.up_proj.weight, -s, s)
            nn.init.zeros_(layer.ffn.down_proj.weight)
        
        # Recompute rotary embeddings on correct device
        head_dim = self.config.n_embd // self.config.n_head
        device = self.tok_embeddings.weight.device
        cos, sin = precompute_freqs_cis(head_dim, self.rotary_seq_len, device=device)
        self.cos, self.sin = cos, sin
        
        # Cast embeddings to bf16 on CUDA
        if self.tok_embeddings.weight.device.type == "cuda":
            self.tok_embeddings.to(dtype=torch.bfloat16)
    
    def _compute_window_sizes(self, config: LlamaConfig) -> List[Tuple[int, int]]:
        """Compute per-layer window sizes for sliding window attention."""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"
        
        long_window = config.sequence_len
        short_window = long_window // 2
        
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes
    
    def get_device(self) -> torch.device:
        return self.tok_embeddings.weight.device
    
    def estimate_flops(self) -> int:
        """
        Estimate FLOPs per token for forward + backward pass.
        
        Each matmul contributes 6 FLOPs per parameter (2 forward, 4 backward).
        Plus attention matrix operations.
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude embeddings (they're lookups, not matmuls)
        nparams_exclude = self.tok_embeddings.weight.numel()
        
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        
        # Attention FLOPs per layer
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token
    
    def num_scaling_params(self) -> dict:
        """Return parameter counts for scaling law analysis."""
        tok_emb = self.tok_embeddings.weight.numel()
        output = self.output.weight.numel()
        layers = sum(p.numel() for layer in self.layers for p in layer.parameters())
        total = tok_emb + output + layers
        
        assert total == sum(p.numel() for p in self.parameters())
        
        return {
            'tok_embeddings': tok_emb,
            'output': output,
            'transformer_layers': layers,
            'total': total,
        }
    
    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        adam_betas: Tuple[float, float] = (0.8, 0.95),
    ):
        """Set up Muon + AdamW optimizer."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate parameter groups
        matrix_params = list(p for layer in self.layers for p in layer.parameters())
        embedding_params = list(self.tok_embeddings.parameters())
        output_params = list(self.output.parameters())
        
        # Scale LR by model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling LR by 1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        param_groups = [
            # AdamW for embeddings and output
            dict(kind='adamw', params=output_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        
        # Muon for matrix params (grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache=None,
        loss_reduction: str = 'mean',
    ):
        """
        Forward pass.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for training (B, T)
            kv_cache: KV cache for inference
            loss_reduction: 'mean' or 'none' for loss computation
        
        Returns:
            loss if targets provided, else logits
        """
        B, T = idx.size()
        device = idx.device
        
        # Get rotary embeddings for current sequence
        assert T <= self.cos.size(1), f"Sequence too long: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        
        # Handle KV cache position offset
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # Embed tokens
        x = self.tok_embeddings(idx)
        x = rms_norm(x, self.config.norm_eps)
        
        # Forward through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, cos_sin, self.window_sizes[i], kv_cache)
        
        # Final norm
        x = rms_norm(x, self.config.norm_eps)
        
        # Compute logits
        # Note: We compute logits for the padded vocabulary and then slice to remove padding.
        # This is more efficient than having a separate output layer for the exact vocab size
        # because it allows tensor core alignment and DDP efficiency.
        softcap = 15.0  # Smooth logit capping
        logits = self.output(x)
        logits = logits[..., :self.config.vocab_size]  # Remove padding
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        if targets is not None:
            # Training: compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # Inference: return logits
            return logits
    
    @torch.inference_mode()
    def generate(
        self,
        tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Simple autoregressive generation (streaming).
        
        Args:
            tokens: Initial token sequence
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling
            seed: Random seed
        
        Yields:
            Generated tokens one at a time
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


# -----------------------------------------------------------------------------
# Convenience functions

def create_model(depth: int, **kwargs) -> Llama:
    """Create a Llama model with the given depth (number of layers)."""
    config = get_config_for_depth(depth)
    # Override any config values from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return Llama(config)
