#!/usr/bin/env python3
"""
Convert nanollama checkpoint to GGUF format for Yent engine inference.

nanollama → GGUF with full compatibility for the Yent Go inference engine.
Handles all architectural differences:
  1. Parameterless RMSNorm → injects identity (all-ones) norm weights
  2. QK-norm flag → metadata for engine to apply RMSNorm on Q/K after RoPE
  3. Conjugate RoPE flag → metadata for engine to use correct rotation convention
  4. SentencePiece tokenizer → embedded in GGUF for self-contained model file

Weight name mapping (nanollama → GGUF):
  tok_embeddings.weight         → token_embd.weight
  layers.N.attn.c_q.weight      → blk.N.attn_q.weight
  layers.N.attn.c_k.weight      → blk.N.attn_k.weight
  layers.N.attn.c_v.weight      → blk.N.attn_v.weight
  layers.N.attn.c_proj.weight   → blk.N.attn_output.weight
  layers.N.ffn.gate_proj.weight → blk.N.ffn_gate.weight
  layers.N.ffn.up_proj.weight   → blk.N.ffn_up.weight
  layers.N.ffn.down_proj.weight → blk.N.ffn_down.weight
  output.weight                 → output.weight
  (generated)                   → blk.N.attn_norm.weight  (all-ones, F32)
  (generated)                   → blk.N.ffn_norm.weight   (all-ones, F32)
  (generated)                   → output_norm.weight       (all-ones, F32)

Usage:
    python scripts/export_gguf.py \\
        --checkpoint weights/micro-yent-bf16.pt \\
        --tokenizer weights/tokenizer.model \\
        --output weights/micro-yent-f16.gguf \\
        --dtype f16

No numpy dependency — uses torch storage directly.
"""

import os
import struct
import argparse
from typing import Dict, List, Tuple, Any

import torch


# ── GGUF constants ──────────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor data types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


# ── Tensor utilities (no numpy) ────────────────────────────────────────────

def tensor_to_bytes(tensor: torch.Tensor, target_dtype: torch.dtype) -> bytes:
    """Convert tensor to raw bytes in target dtype. No numpy required."""
    t = tensor.to(target_dtype).contiguous().clone()
    nbytes = t.nelement() * t.element_size()
    return bytes(t.untyped_storage())[:nbytes]


# ── GGUF writer ─────────────────────────────────────────────────────────────

class GGUFWriter:
    """Writes a valid GGUF v3 file. No numpy dependency."""

    def __init__(self, path: str):
        self.path = path
        self.kv_pairs: List[Tuple[str, int, Any]] = []
        # (name, raw_bytes, ggml_type, shape_tuple)
        self.tensors: List[Tuple[str, bytes, int, Tuple[int, ...]]] = []

    def add_uint32(self, key: str, value: int):
        self.kv_pairs.append((key, GGUF_TYPE_UINT32, value))

    def add_int32(self, key: str, value: int):
        self.kv_pairs.append((key, GGUF_TYPE_INT32, value))

    def add_float32(self, key: str, value: float):
        self.kv_pairs.append((key, GGUF_TYPE_FLOAT32, value))

    def add_bool(self, key: str, value: bool):
        self.kv_pairs.append((key, GGUF_TYPE_BOOL, value))

    def add_string(self, key: str, value: str):
        self.kv_pairs.append((key, GGUF_TYPE_STRING, value))

    def add_string_array(self, key: str, values: List[str]):
        self.kv_pairs.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, values)))

    def add_float32_array(self, key: str, values: List[float]):
        self.kv_pairs.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, values)))

    def add_int32_array(self, key: str, values: List[int]):
        self.kv_pairs.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, values)))

    def add_tensor_raw(self, name: str, raw_bytes: bytes, ggml_type: int, shape: Tuple[int, ...]):
        """Add a tensor from raw bytes with explicit shape."""
        self.tensors.append((name, raw_bytes, ggml_type, shape))

    def add_tensor(self, name: str, tensor: torch.Tensor, ggml_type: int):
        """Add a tensor from a torch tensor."""
        if ggml_type == GGML_TYPE_F16:
            raw = tensor_to_bytes(tensor, torch.float16)
        else:
            raw = tensor_to_bytes(tensor, torch.float32)
        self.tensors.append((name, raw, ggml_type, tuple(tensor.shape)))

    def _write_string(self, f, s: str):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_kv(self, f, key: str, vtype: int, value):
        self._write_string(f, key)
        f.write(struct.pack('<I', vtype))

        if vtype == GGUF_TYPE_UINT8:
            f.write(struct.pack('<B', value))
        elif vtype == GGUF_TYPE_INT8:
            f.write(struct.pack('<b', value))
        elif vtype == GGUF_TYPE_UINT16:
            f.write(struct.pack('<H', value))
        elif vtype == GGUF_TYPE_INT16:
            f.write(struct.pack('<h', value))
        elif vtype == GGUF_TYPE_UINT32:
            f.write(struct.pack('<I', value))
        elif vtype == GGUF_TYPE_INT32:
            f.write(struct.pack('<i', value))
        elif vtype == GGUF_TYPE_FLOAT32:
            f.write(struct.pack('<f', value))
        elif vtype == GGUF_TYPE_BOOL:
            f.write(struct.pack('<B', 1 if value else 0))
        elif vtype == GGUF_TYPE_STRING:
            self._write_string(f, value)
        elif vtype == GGUF_TYPE_UINT64:
            f.write(struct.pack('<Q', value))
        elif vtype == GGUF_TYPE_INT64:
            f.write(struct.pack('<q', value))
        elif vtype == GGUF_TYPE_FLOAT64:
            f.write(struct.pack('<d', value))
        elif vtype == GGUF_TYPE_ARRAY:
            elem_type, elements = value
            f.write(struct.pack('<I', elem_type))
            f.write(struct.pack('<Q', len(elements)))
            for elem in elements:
                if elem_type == GGUF_TYPE_STRING:
                    self._write_string(f, elem)
                elif elem_type == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack('<f', elem))
                elif elem_type == GGUF_TYPE_INT32:
                    f.write(struct.pack('<i', elem))
                elif elem_type == GGUF_TYPE_UINT32:
                    f.write(struct.pack('<I', elem))

    def _align(self, f, alignment: int = GGUF_ALIGNMENT):
        pos = f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        if pad > 0:
            f.write(b'\x00' * pad)

    def write(self):
        """Write the complete GGUF file."""
        with open(self.path, 'wb') as f:
            # ── Header ──
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.kv_pairs)))

            # ── Metadata KV pairs ──
            for key, vtype, value in self.kv_pairs:
                self._write_kv(f, key, vtype, value)

            # ── Tensor info ──
            # Pre-compute offsets within the data section
            data_offset = 0
            tensor_offsets = []
            for i, (name, raw, ggml_type, shape) in enumerate(self.tensors):
                # Align within data section
                if i > 0:
                    data_offset = ((data_offset + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT
                tensor_offsets.append(data_offset)
                data_offset += len(raw)

            for i, (name, raw, ggml_type, shape) in enumerate(self.tensors):
                self._write_string(f, name)
                ndims = len(shape)
                f.write(struct.pack('<I', ndims))
                # GGML dimensions are reversed from PyTorch (innermost first)
                for d in reversed(shape):
                    f.write(struct.pack('<Q', d))
                f.write(struct.pack('<I', ggml_type))
                f.write(struct.pack('<Q', tensor_offsets[i]))

            # ── Alignment padding before data section ──
            self._align(f)

            # ── Tensor data ──
            for i, (name, raw, ggml_type, shape) in enumerate(self.tensors):
                self._align(f)
                f.write(raw)

        file_size = os.path.getsize(self.path)
        print(f"  Written: {self.path} ({file_size / 1024 / 1024:.2f} MB)")
        print(f"  Tensors: {len(self.tensors)}")
        print(f"  Metadata KV: {len(self.kv_pairs)}")


# ── Weight mapping ──────────────────────────────────────────────────────────

WEIGHT_MAP = {
    "tok_embeddings.weight": "token_embd.weight",
    "output.weight": "output.weight",
}

LAYER_WEIGHT_MAP = {
    "attn.c_q.weight":      "attn_q.weight",
    "attn.c_k.weight":      "attn_k.weight",
    "attn.c_v.weight":      "attn_v.weight",
    "attn.c_proj.weight":   "attn_output.weight",
    "ffn.gate_proj.weight": "ffn_gate.weight",
    "ffn.up_proj.weight":   "ffn_up.weight",
    "ffn.down_proj.weight": "ffn_down.weight",
}


def map_name(name: str) -> str:
    """Map nanollama weight name to GGUF tensor name."""
    if name in WEIGHT_MAP:
        return WEIGHT_MAP[name]
    if name.startswith("layers."):
        parts = name.split(".", 2)
        layer_idx = parts[1]
        rest = parts[2]
        if rest in LAYER_WEIGHT_MAP:
            return f"blk.{layer_idx}.{LAYER_WEIGHT_MAP[rest]}"
    raise ValueError(f"Unknown weight name: {name}")


def compute_intermediate_size(n_embd: int, multiple_of: int = 256) -> int:
    """Compute SwiGLU intermediate size matching nanollama's formula."""
    hidden = int(2 * (4 * n_embd) / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


# ── Tokenizer extraction ───────────────────────────────────────────────────

def load_tokenizer_metadata(tokenizer_path: str) -> Dict[str, Any]:
    """Extract tokenizer info from SentencePiece model for GGUF embedding."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("WARNING: sentencepiece not installed, skipping tokenizer embedding")
        return {}

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = sp.get_piece_size()

    tokens = []
    scores = []
    token_types = []

    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        tokens.append(piece)
        scores.append(float(score))

        if sp.is_unknown(i):
            token_types.append(2)
        elif sp.is_control(i):
            token_types.append(3)
        elif sp.is_byte(i):
            token_types.append(6)
        elif sp.is_unused(i):
            token_types.append(5)
        else:
            token_types.append(1)

    return {
        "model": "llama",
        "tokens": tokens,
        "scores": scores,
        "token_types": token_types,
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "vocab_size": vocab_size,
    }


# ── Main conversion ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert nanollama checkpoint to GGUF for Yent engine"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to nanollama checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to SentencePiece .model file (optional, embeds tokenizer)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output GGUF file path")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f32", "f16"],
                        help="Weight dtype (default: f16)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("nanollama → GGUF converter")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer:  {args.tokenizer or '(not embedded)'}")
    print(f"Output:     {args.output}")
    print(f"Dtype:      {args.dtype}")
    print()

    # ── Load checkpoint ──
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    print(f"  Weights: {len(state)} tensors")
    if config:
        print(f"  Config:  {config}")

    # ── Infer model config ──
    n_embd = config.get("n_embd", 0)
    n_layer = config.get("n_layer", 0)
    n_head = config.get("n_head", 0)
    n_kv_head = config.get("n_kv_head", 0)
    vocab_size = config.get("vocab_size", 0)
    sequence_len = config.get("sequence_len", 2048)
    norm_eps = config.get("norm_eps", 1e-5)
    rope_theta = config.get("rope_theta", 500000.0)

    if n_embd == 0 and "tok_embeddings.weight" in state:
        n_embd = state["tok_embeddings.weight"].shape[1]
    if vocab_size == 0 and "tok_embeddings.weight" in state:
        vocab_size = state["tok_embeddings.weight"].shape[0]
    if n_layer == 0:
        layer_indices = set()
        for k in state:
            if k.startswith("layers."):
                layer_indices.add(int(k.split(".")[1]))
        n_layer = max(layer_indices) + 1 if layer_indices else 0
    if n_head == 0 and "layers.0.attn.c_q.weight" in state:
        q_out = state["layers.0.attn.c_q.weight"].shape[0]
        head_dim = n_embd // (q_out // n_embd) if n_embd else 64
        n_head = q_out // head_dim if head_dim else 8
    if n_kv_head == 0 and "layers.0.attn.c_k.weight" in state:
        k_out = state["layers.0.attn.c_k.weight"].shape[0]
        head_dim = n_embd // n_head if n_head else 64
        n_kv_head = k_out // head_dim if head_dim else 2

    head_dim = n_embd // n_head if n_head else 64
    intermediate_size = compute_intermediate_size(n_embd)

    print(f"\n  Model: {n_layer}L / {n_embd}D / {n_head}H / {n_kv_head}KV / {head_dim}HD")
    print(f"  Vocab: {vocab_size}, Seq: {sequence_len}, FFN: {intermediate_size}")
    print(f"  RoPE theta: {rope_theta}, Norm eps: {norm_eps}")

    # ── Prepare GGUF writer ──
    writer = GGUFWriter(args.output)

    # ── Architecture metadata ──
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", "nanollama-micro-yent")
    writer.add_uint32("llama.block_count", n_layer)
    writer.add_uint32("llama.embedding_length", n_embd)
    writer.add_uint32("llama.attention.head_count", n_head)
    writer.add_uint32("llama.attention.head_count_kv", n_kv_head)
    writer.add_uint32("llama.feed_forward_length", intermediate_size)
    writer.add_uint32("llama.context_length", sequence_len)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", norm_eps)
    writer.add_float32("llama.rope.freq_base", rope_theta)
    writer.add_uint32("llama.vocab_size", vocab_size)

    # nanollama-specific flags for Yent engine
    writer.add_bool("nanollama.qk_norm", True)
    writer.add_bool("nanollama.rope_conjugate", True)

    # ── Tokenizer metadata ──
    tok_meta = {}
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"\nLoading tokenizer: {args.tokenizer}")
        tok_meta = load_tokenizer_metadata(args.tokenizer)

    if tok_meta:
        writer.add_string("tokenizer.ggml.model", tok_meta["model"])
        writer.add_string_array("tokenizer.ggml.tokens", tok_meta["tokens"])
        writer.add_float32_array("tokenizer.ggml.scores", tok_meta["scores"])
        writer.add_int32_array("tokenizer.ggml.token_type", tok_meta["token_types"])
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok_meta["bos_id"])
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok_meta["eos_id"])
        print(f"  Tokenizer: {tok_meta['vocab_size']} pieces embedded")
    else:
        writer.add_string("tokenizer.ggml.model", "llama")
        print("  Tokenizer: not embedded (load separately)")

    # ── Convert weight tensors ──
    ggml_type = GGML_TYPE_F16 if args.dtype == "f16" else GGML_TYPE_F32
    torch_dtype = torch.float16 if args.dtype == "f16" else torch.float32

    print(f"\nConverting weights to {args.dtype}...")
    converted = 0
    for name in sorted(state.keys()):
        tensor = state[name]
        gguf_name = map_name(name)
        writer.add_tensor(gguf_name, tensor.float(), ggml_type)
        shape_str = "x".join(str(d) for d in tensor.shape)
        print(f"  {name:45s} → {gguf_name:35s}  [{shape_str}]")
        converted += 1

    # ── Inject identity RMSNorm weights ──
    print(f"\nInjecting identity norm weights (all-ones, F32)...")
    ones = torch.ones(n_embd, dtype=torch.float32)

    for i in range(n_layer):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", ones, GGML_TYPE_F32)
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", ones, GGML_TYPE_F32)
        print(f"  blk.{i}.attn_norm.weight  [{n_embd}] = ones")
        print(f"  blk.{i}.ffn_norm.weight   [{n_embd}] = ones")

    writer.add_tensor("output_norm.weight", ones, GGML_TYPE_F32)
    print(f"  output_norm.weight        [{n_embd}] = ones")

    total_tensors = converted + n_layer * 2 + 1
    print(f"\nTotal: {total_tensors} tensors ({converted} from checkpoint + {n_layer * 2 + 1} injected norms)")

    # ── Write GGUF ──
    print(f"\nWriting GGUF...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer.write()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
