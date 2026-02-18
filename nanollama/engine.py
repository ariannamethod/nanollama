"""
Inference engine for nanollama with GQA-optimized KV cache.

GQA (Grouped Query Attention) means fewer KV heads than query heads,
making the KV cache more memory efficient during inference.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from typing import List, Tuple, Optional, Generator

from nanollama.common import compute_init, autodetect_device_type
from nanollama.checkpoint_manager import load_model
from contextlib import nullcontext


# -----------------------------------------------------------------------------
# Calculator tool helpers

@contextmanager
def timeout(duration: int, formula: str):
    """Timeout context manager for safe eval."""
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula: str, max_time: int = 3):
    """Safely evaluate a formula with timeout."""
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr: str):
    """
    Evaluate a Python expression safely.
    Supports math expressions and string operations like .count()
    """
    expr = expr.replace(",", "")
    
    # Pure math expression
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)
    
    # String operations
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed_chars for x in expr):
        return None
    
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None
    
    if '.count(' not in expr:
        return None
    
    return eval_with_timeout(expr)


# -----------------------------------------------------------------------------
# KV Cache for GQA

class KVCache:
    """
    KV Cache optimized for Grouped Query Attention (GQA).
    
    In GQA, we have fewer KV heads than query heads, making the cache
    more memory efficient. This is a key advantage of Llama 3 over GPT-2.
    
    Cache shape: (n_layers, B, T, n_kv_heads, head_dim)
    
    Note: n_kv_heads < n_heads for GQA, saving memory!
    """
    
    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_kv_heads = num_kv_heads  # Note: This is KV heads, not query heads!
        self.head_dim = head_dim
        
        # Pre-allocate cache tensors: (n_layers, B, T, n_kv_heads, D)
        self.k_cache = torch.zeros(
            num_layers, batch_size, seq_len, num_kv_heads, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_layers, batch_size, seq_len, num_kv_heads, head_dim,
            device=device, dtype=dtype
        )
        
        # Current sequence length per batch element
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
    
    def get_pos(self) -> int:
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()
    
    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def advance(self, num_tokens: int):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens
    
    def prefill(self, other: 'KVCache'):
        """
        Copy cached KV from another cache into this one.
        Used for batch=1 prefill followed by parallel sample generation.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers
        assert self.n_kv_heads == other.n_kv_heads
        assert self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
    
    def memory_usage_bytes(self) -> int:
        """Return memory usage of the cache in bytes."""
        return self.k_cache.numel() * self.k_cache.element_size() * 2


# -----------------------------------------------------------------------------
# Sampling helpers

@torch.inference_mode()
def sample_next_token(
    logits: torch.Tensor,
    rng: Optional[torch.Generator],
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Sample a single next token from logits of shape (B, vocab_size)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------
# Row state for generation

class RowState:
    """Per-row state tracking during generation."""
    
    def __init__(self, current_tokens: Optional[List[int]] = None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


# -----------------------------------------------------------------------------
# Inference Engine

class Engine:
    """
    Efficient inference engine for nanollama models.
    
    Features:
    - GQA-optimized KV cache (smaller memory footprint)
    - Streaming token generation
    - Tool use support (calculator)
    - Batch generation with shared prefill
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.inference_mode()
    def generate(
        self,
        tokens: List[int],
        num_samples: int = 1,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate tokens with streaming output.
        
        Args:
            tokens: Input token sequence
            num_samples: Number of parallel samples to generate
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed
        
        Yields:
            (token_column, token_masks): Lists of generated tokens and masks
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        
        # Get special tokens for tool use
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        eot = get_special("<|eot_id|>")  # Llama 3 end of turn
        bos = self.tokenizer.get_bos_token_id()
        
        # Determine which tokens signal completion
        end_tokens = {assistant_end, eot, bos}
        end_tokens.discard(None)
        
        # KV cache parameters for GQA
        m = self.model.config
        kv_model_kwargs = {
            "num_kv_heads": m.n_kv_head,  # GQA: fewer KV heads!
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        
        # 1) Batch-1 prefill
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)
        
        # 2) Replicate KV cache for each sample
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill
        
        # 3) Initialize row states
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        
        # 4) Main generation loop
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break
            
            # Sample next tokens
            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()
            
            # Process each row
            token_column = []
            token_masks = []
            
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                
                state.current_tokens.append(next_token)
                
                # Check for completion
                if next_token in end_tokens:
                    state.completed = True
                
                # Handle tool use
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
            
            yield token_column, token_masks
            num_generated += 1
            
            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
    
    def generate_batch(
        self,
        tokens: List[int],
        num_samples: int = 1,
        **kwargs
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Non-streaming batch generation.
        
        Returns:
            (results, masks): Lists of token sequences and their masks
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        eot = self.tokenizer.encode_special("<|eot_id|>")
        bos = self.tokenizer.get_bos_token_id()
        
        end_tokens = {assistant_end, eot, bos}
        end_tokens.discard(None)
        
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in end_tokens:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        
        return results, masks


# -----------------------------------------------------------------------------
# Main test

if __name__ == "__main__":
    """Quick test of naive vs engine generation."""
    import time
    
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    
    # Load model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    
    kwargs = dict(max_tokens=64, temperature=0.0)
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    
    # Reference generation
    print("Reference generation:")
    generated_tokens = []
    torch.cuda.synchronize() if device_type == "cuda" else None
    t0 = time.time()
    with autocast_ctx:
        for token in model.generate(prompt_tokens, **kwargs):
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize() if device_type == "cuda" else None
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    
    # Engine generation
    print("\nEngine generation:")
    generated_tokens = []
    engine = Engine(model, tokenizer)
    torch.cuda.synchronize() if device_type == "cuda" else None
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in engine.generate(prompt_tokens, num_samples=1, **kwargs):
            token = token_column[0]
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize() if device_type == "cuda" else None
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    
    # Compare
    match = reference_ids == generated_tokens
    print(f"Match: {match}")
