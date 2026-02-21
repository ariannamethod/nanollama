"""
Llama 3 model for nanollama.
RoPE, GQA/MHA, SwiGLU, RMSNorm, pre-norm, no bias, optional tied embeddings.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanollama.common import get_dist_info, print0
from nanollama.optim import MuonAdamW, DistMuonAdamW

@dataclass
class LlamaConfig:
    """Llama 3 model configuration."""
    sequence_len: int = 2048
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    norm_eps: float = 1e-5
    multiple_of: int = 256
    window_pattern: str = "SSSL"
    rope_theta: float = 10000.0   # 10000 for short context (2048); increase for longer
    tie_embeddings: bool = False   # share tok_embeddings and output weights

# Explicit named configs — calculated for optimal depth/width ratio per model size.
# Design: deep-and-thin (MobileLLM insight), tied embeddings for small models,
# head_dim=64, MHA for nano/micro, GQA for mini+.
NAMED_CONFIGS = {
    #  name      depth  dim   heads  kv_heads  tied     ~params
    "nano":   (  12,   384,    6,      6,    False),  #  45.8M (untied: fixes 50x LR mismatch on output head)
    "micro":  (  16,   512,    8,      8,    False),  #  87.3M (untied: avoids 50x LR mismatch + loss 23 start)
    "mini":   (  20,   768,   12,      4,    False),  # 175.0M (untied: same reason)
    "small":  (  24,  1024,   16,      4,    False),  # 338M
    "goldie": (  22,  2048,   32,      8,    False),  # 1.1B
    "medium": (  32,  2048,   32,      8,    False),  # 1.6B
    "large":  (  36,  3072,   48,      8,    False),  # 3.7B
    "big":    (  38,  4096,   64,     16,    False),  # 7.0B
}

def get_named_config(name: str) -> LlamaConfig:
    """Get config for a named model size. Preferred over get_config_for_depth."""
    if name not in NAMED_CONFIGS:
        raise ValueError(f"Unknown model size '{name}'. Available: {list(NAMED_CONFIGS.keys())}")
    depth, n_embd, n_head, n_kv_head, tied = NAMED_CONFIGS[name]
    return LlamaConfig(n_layer=depth, n_embd=n_embd, n_head=n_head,
                       n_kv_head=n_kv_head, tie_embeddings=tied)

def get_config_for_depth(depth: int) -> LlamaConfig:
    """Fallback config from depth. Use get_named_config() for production training."""
    n_embd = max(384, 64 * (depth * 48 // 64))  # ~48 per layer, rounded to 64
    n_embd = 64 * (n_embd // 64)
    n_head = n_embd // 64  # head_dim = 64
    if n_embd <= 512:
        n_kv_head = n_head  # MHA for small models
    else:
        # GQA: ensure n_head % n_kv_head == 0
        n_kv_head = max(4, n_head // 4)
        while n_head % n_kv_head != 0:
            n_kv_head -= 1
        n_kv_head = max(1, n_kv_head)
    tied = n_embd <= 768
    return LlamaConfig(n_layer=depth, n_embd=n_embd, n_head=n_head,
                       n_kv_head=n_kv_head, tie_embeddings=tied)

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Parameterless RMS norm — used only for QK-norm (standard in Llama 3)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale — standard Llama 3, llama.cpp compatible."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 500000.0,
                          device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin. Llama 3 uses theta=500000 (NOT 10000 like Llama 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)[None, :, None, :]
    sin = freqs.sin().to(torch.bfloat16)[None, :, None, :]
    return cos, sin

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x (B, T, H, D)."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)


class CausalSelfAttention(nn.Module):
    """GQA: fewer KV heads than query heads for efficient inference."""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.n_embd, self.head_dim = config.n_embd, config.n_embd // config.n_head
        self.norm_eps = config.norm_eps
        self.n_rep = self.n_head // self.n_kv_head
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor],
                window_size: Tuple[int, int], kv_cache=None) -> torch.Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q, self.norm_eps), rms_norm(k, self.norm_eps)

        if kv_cache is not None:
            y = self._attn_with_cache(q, k, v, kv_cache)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        else:
            y = self._attention(q, k, v)
        
        return self.c_proj(y.contiguous().view(B, T, -1))
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return y.transpose(1, 2)
    
    def _attn_with_cache(self, q, k, v, kv_cache) -> torch.Tensor:
        B, T = q.shape[:2]
        pos = kv_cache.get_pos()
        k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
        k_cache[:, pos:pos+T], v_cache[:, pos:pos+T] = k, v
        k_full, v_full = k_cache[:, :pos+T], v_cache[:, :pos+T]
        if self.n_rep > 1:
            k_full = k_full.repeat_interleave(self.n_rep, dim=2)
            v_full = v_full.repeat_interleave(self.n_rep, dim=2)
        q, k_full, v_full = q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k_full, v_full, dropout_p=0.0, is_causal=True)
        return y.transpose(1, 2)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: down(swish(gate(x)) * up(x))"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm→attn→residual, norm→ffn→residual"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor],
                window_size: Tuple[int, int], kv_cache=None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos_sin, window_size, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Llama(nn.Module):
    """Llama 3 model for training FROM SCRATCH (not a wrapper around Meta weights)."""
    
    def __init__(self, config: LlamaConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        
        # Pad vocab for tensor core efficiency
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab {config.vocab_size} → {padded_vocab}")
        
        self.tok_embeddings = nn.Embedding(padded_vocab, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        if config.tie_embeddings:
            self.output = None  # will use tok_embeddings.weight in forward
        else:
            self.output = nn.Linear(config.n_embd, padded_vocab, bias=False)

        # ResFormer per-layer scalars (nanochat convention)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, self.rotary_seq_len, theta=config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    @torch.no_grad()
    def init_weights(self):
        """Initialize weights following Llama conventions."""
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)
        if self.output is not None:
            nn.init.normal_(self.output.weight, mean=0.0, std=0.001)
        s = (3 ** 0.5) * (self.config.n_embd ** -0.5)
        for layer in self.layers:
            layer.attn_norm.weight.fill_(1.0)
            layer.ffn_norm.weight.fill_(1.0)
            nn.init.uniform_(layer.attn.c_q.weight, -s, s)
            nn.init.uniform_(layer.attn.c_k.weight, -s, s)
            nn.init.uniform_(layer.attn.c_v.weight, -s, s)
            nn.init.zeros_(layer.attn.c_proj.weight)
            nn.init.uniform_(layer.ffn.gate_proj.weight, -s, s)
            nn.init.uniform_(layer.ffn.up_proj.weight, -s, s)
            nn.init.zeros_(layer.ffn.down_proj.weight)
        self.norm.weight.fill_(1.0)

        # ResFormer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        head_dim = self.config.n_embd // self.config.n_head
        device = self.tok_embeddings.weight.device
        self.cos, self.sin = precompute_freqs_cis(head_dim, self.rotary_seq_len, 
                                                   theta=self.config.rope_theta, device=device)
        if device.type == "cuda":
            self.tok_embeddings.to(dtype=torch.bfloat16)
    
    def _compute_window_sizes(self, config: LlamaConfig) -> List[Tuple[int, int]]:
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        windows = [(long_w, 0) if c == "L" else (short_w, 0) for c in pattern * config.n_layer]
        windows = windows[:config.n_layer]
        windows[-1] = (long_w, 0)
        return windows
    
    def get_device(self) -> torch.device:
        return self.tok_embeddings.weight.device
    
    def estimate_flops(self) -> int:
        # Embedding lookup is free (row select, not matmul), so subtract its params.
        # But with tied embeddings, the same weight IS used in output matmul — don't subtract.
        if self.output is not None:
            nparams = sum(p.numel() for p in self.parameters()) - self.tok_embeddings.weight.numel()
        else:
            nparams = sum(p.numel() for p in self.parameters())
        # Exclude scalar params (not matmuls)
        nparams -= self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(w[0], t) for w in self.window_sizes)
        return 6 * nparams + attn_flops
    
    def num_scaling_params(self) -> dict:
        tok = self.tok_embeddings.weight.numel()
        out = self.output.weight.numel() if self.output is not None else 0
        layers = sum(p.numel() for l in self.layers for p in l.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        return {'tok_embeddings': tok, 'output': out, 'transformer_layers': layers,
                'scalars': scalars, 'total': tok + out + layers + scalars,
                'tied': self.output is None}
    
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        ddp, rank, local_rank, world_size = get_dist_info()
        # AdamW LR scales with 1/sqrt(dmodel), tuned at 768 (nanochat convention)
        adamw_scale = (self.config.n_embd / 768) ** -0.5

        # Separate 1D (norms) from 2D (matrices) in transformer layers
        matrix_params = []
        norm_params = list(self.norm.parameters())  # output norm
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() == 1:
                    norm_params.append(p)
                else:
                    matrix_params.append(p)

        param_groups = []
        if self.output is not None:
            param_groups.append(dict(kind='adamw', params=list(self.output.parameters()),
                                     lr=unembedding_lr * adamw_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind='adamw', params=list(self.tok_embeddings.parameters()),
                                     lr=embedding_lr * adamw_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        else:
            param_groups.append(dict(kind='adamw', params=list(self.tok_embeddings.parameters()),
                                     lr=embedding_lr * adamw_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        param_groups.append(dict(kind='adamw', params=norm_params, lr=unembedding_lr * adamw_scale,
                                 betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # ResFormer scalars
        param_groups.append(dict(kind='adamw', params=[self.resid_lambdas],
                                 lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        param_groups.append(dict(kind='adamw', params=[self.x0_lambdas],
                                 lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups: matrix_lr NOT scaled by dmodel (nanochat convention)
        for shape in sorted({p.shape for p in matrix_params}):
            group = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))

        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]
        return optimizer
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_cache=None, loss_reduction: str = 'mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.tok_embeddings(idx)
        x = rms_norm(x)  # normalize after embedding (nanochat convention)
        x0 = x  # save for ResFormer x0 residual
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = layer(x, cos_sin, self.window_sizes[i], kv_cache)
        x = self.norm(x)

        if self.output is not None:
            logits = self.output(x)[..., :self.config.vocab_size].float()
        else:
            logits = F.linear(x, self.tok_embeddings.weight)[..., :self.config.vocab_size].float()

        # Logit softcap (nanochat convention, replaces z-loss)
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits
    
    @torch.inference_mode()
    def generate(self, tokens: List[int], max_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, seed: int = 42):
        device = self.get_device()
        rng = torch.Generator(device=device).manual_seed(seed) if temperature > 0 else None
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        autocast = torch.amp.autocast(device.type, dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

        for _ in range(max_tokens):
            with autocast:
                logits = self.forward(ids)[:, -1, :]
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            yield next_id.item()

def create_model(depth: int, **kwargs) -> Llama:
    config = get_config_for_depth(depth)
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return Llama(config)
