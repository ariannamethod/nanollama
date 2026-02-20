#!/usr/bin/env python3
"""Quantize GGUF F16 → Q8_0. No torch, no numpy required.

Reads an existing GGUF file (F16 or F32 weights), quantizes 2D tensors
to Q8_0, keeps 1D norms as F32. Writes a new GGUF file.

Usage:
    python3 scripts/quantize_gguf.py weights/micro-yent-f16.gguf weights/micro-yent-q8_0.gguf
"""

import struct
import sys
import os
import math

# ── GGUF/GGML constants ─────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747
GGUF_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

# Type sizes
GGML_TYPE_SIZE = {
    GGML_TYPE_F32: 4,  # 4 bytes per element
    GGML_TYPE_F16: 2,  # 2 bytes per element
    GGML_TYPE_Q8_0: None,  # block-quantized, special
}

Q8_BLOCK_SIZE = 32
Q8_BYTES_PER_BLOCK = 34  # 2 (fp16 scale) + 32 (int8 values)

# GGUF metadata types
GGUF_TYPES = {
    0: ('uint8', '<B', 1),
    1: ('int8', '<b', 1),
    2: ('uint16', '<H', 2),
    3: ('int16', '<h', 2),
    4: ('uint32', '<I', 4),
    5: ('int32', '<i', 4),
    6: ('float32', '<f', 4),
    7: ('bool', '<B', 1),
    8: ('string', None, None),
    9: ('array', None, None),
    10: ('uint64', '<Q', 8),
    11: ('int64', '<q', 8),
    12: ('float64', '<d', 8),
}


# ── FP16 ↔ FP32 conversion (pure Python) ────────────────────────────────

def half_to_float(h):
    """Convert IEEE 754 half-precision (uint16) to float."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF

    if exp == 0:
        if frac == 0:
            return (-1) ** sign * 0.0
        # Denormalized
        return (-1) ** sign * 2 ** (-14) * (frac / 1024.0)
    elif exp == 31:
        if frac == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        return (-1) ** sign * 2 ** (exp - 15) * (1.0 + frac / 1024.0)


def float_to_half(f):
    """Convert float to IEEE 754 half-precision (uint16)."""
    # Use struct to convert float→half via intermediate
    # Python 3.6+ supports 'e' format for half-precision
    return struct.unpack('<H', struct.pack('<e', f))[0]


# ── GGUF Reader ──────────────────────────────────────────────────────────

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_value(f, vtype):
    if vtype == 8:  # string
        return read_string(f)
    elif vtype == 9:  # array
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, elem_type) for _ in range(count)]
    else:
        name, fmt, size = GGUF_TYPES[vtype]
        return struct.unpack(fmt, f.read(size))[0]


def read_gguf(path):
    """Read GGUF file, return (metadata_dict, tensor_infos, header_bytes_count)."""
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic={magic:#x})"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        print(f"  GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")

        # Read metadata
        metadata = []
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            metadata.append((key, vtype, value))

        # Read tensor infos
        tensor_infos = []
        for _ in range(n_tensors):
            name = read_string(f)
            ndims = struct.unpack('<I', f.read(4))[0]
            # GGML dimensions are reversed (innermost first)
            dims = []
            for _ in range(ndims):
                dims.append(struct.unpack('<Q', f.read(8))[0])
            ggml_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append({
                'name': name,
                'ndims': ndims,
                'dims': dims,  # GGML order (reversed from row-major)
                'type': ggml_type,
                'offset': offset,
            })

        # Data section starts after alignment
        header_end = f.tell()
        data_start = ((header_end + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT

    return metadata, tensor_infos, data_start, version


def compute_tensor_bytes(info):
    """Compute raw byte size of a tensor given its GGML type and dims."""
    # Total elements = product of all dims
    n_elements = 1
    for d in info['dims']:
        n_elements *= d

    t = info['type']
    if t == GGML_TYPE_F32:
        return n_elements * 4
    elif t == GGML_TYPE_F16:
        return n_elements * 2
    elif t == GGML_TYPE_Q8_0:
        nblocks = n_elements // Q8_BLOCK_SIZE
        return nblocks * Q8_BYTES_PER_BLOCK
    else:
        raise ValueError(f"Unsupported type {t} for tensor {info['name']}")


# ── Quantization ─────────────────────────────────────────────────────────

def dequant_f16_to_f32(data, n_elements):
    """Dequantize F16 raw bytes to list of floats."""
    values = []
    for i in range(n_elements):
        h = struct.unpack_from('<H', data, i * 2)[0]
        values.append(half_to_float(h))
    return values


def dequant_f32_to_f32(data, n_elements):
    """Read F32 raw bytes to list of floats."""
    values = []
    for i in range(n_elements):
        values.append(struct.unpack_from('<f', data, i * 4)[0])
    return values


def quantize_to_q8_0(values):
    """Quantize float list to Q8_0 bytes."""
    n = len(values)
    assert n % Q8_BLOCK_SIZE == 0, f"Size {n} not divisible by {Q8_BLOCK_SIZE}"

    out = bytearray()
    nblocks = n // Q8_BLOCK_SIZE

    for b in range(nblocks):
        block = values[b * Q8_BLOCK_SIZE:(b + 1) * Q8_BLOCK_SIZE]

        # Find scale
        amax = max(abs(v) for v in block)
        if amax == 0:
            scale = 1.0
        else:
            scale = amax / 127.0

        inv_scale = 1.0 / scale

        # Quantize
        quants = []
        for v in block:
            q = round(v * inv_scale)
            q = max(-128, min(127, q))
            quants.append(q)

        # Pack: fp16 scale + 32 int8
        out += struct.pack('<e', scale)
        for q in quants:
            out += struct.pack('<b', q)

    return bytes(out)


# ── GGUF Writer ──────────────────────────────────────────────────────────

def write_string(f, s):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_value(f, vtype, value):
    if vtype == 8:  # string
        write_string(f, value)
    elif vtype == 9:  # array
        # value is a list, need to figure out elem_type
        # We stored it as a list during reading, need original elem_type
        # This is handled specially in write_kv
        pass
    else:
        _, fmt, _ = GGUF_TYPES[vtype]
        f.write(struct.pack(fmt, value))


def write_kv(f, key, vtype, value):
    write_string(f, key)
    f.write(struct.pack('<I', vtype))
    if vtype == 8:  # string
        write_string(f, value)
    elif vtype == 9:  # array
        # We need to detect elem_type from the stored value
        # During read, arrays are stored as plain lists
        # We'll write them back by inferring type from the metadata key
        # For simplicity, store the raw bytes during read
        raise NotImplementedError("Array KV re-writing needs raw copy")
    else:
        _, fmt, _ = GGUF_TYPES[vtype]
        f.write(struct.pack(fmt, value))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 quantize_gguf.py <input.gguf> <output.gguf>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"=== GGUF Quantizer: F16/F32 → Q8_0 ===")
    print(f"Input:  {input_path} ({os.path.getsize(input_path) / 1024 / 1024:.1f} MB)")
    print(f"Output: {output_path}")
    print()

    # Read input GGUF structure
    print("Reading GGUF header...")
    metadata, tensor_infos, data_start, version = read_gguf(input_path)

    # Read the raw file for data extraction
    with open(input_path, 'rb') as f:
        raw_file = f.read()

    # Process tensors
    print(f"\nQuantizing {len(tensor_infos)} tensors...")
    new_tensors = []  # (name, raw_bytes, ggml_type, dims)

    for info in tensor_infos:
        name = info['name']
        n_elements = 1
        for d in info['dims']:
            n_elements *= d

        # Extract raw tensor data
        tensor_start = data_start + info['offset']
        tensor_bytes = compute_tensor_bytes(info)
        tensor_data = raw_file[tensor_start:tensor_start + tensor_bytes]

        is_1d = info['ndims'] == 1

        if is_1d:
            # Keep norms as F32
            if info['type'] == GGML_TYPE_F16:
                values = dequant_f16_to_f32(tensor_data, n_elements)
                raw = struct.pack(f'<{n_elements}f', *values)
            elif info['type'] == GGML_TYPE_F32:
                raw = tensor_data
            else:
                raw = tensor_data
            new_tensors.append((name, raw, GGML_TYPE_F32, info['dims']))
            print(f"  {name:40s} [{n_elements}] → F32 (norm)")
        else:
            # Quantize 2D tensors to Q8_0
            if info['type'] == GGML_TYPE_F16:
                values = dequant_f16_to_f32(tensor_data, n_elements)
            elif info['type'] == GGML_TYPE_F32:
                values = dequant_f32_to_f32(tensor_data, n_elements)
            else:
                raise ValueError(f"Cannot quantize type {info['type']}")

            if n_elements % Q8_BLOCK_SIZE != 0:
                print(f"  WARNING: {name} ({n_elements}) not Q8-compatible, keeping F32")
                raw = struct.pack(f'<{n_elements}f', *values)
                new_tensors.append((name, raw, GGML_TYPE_F32, info['dims']))
            else:
                raw = quantize_to_q8_0(values)
                new_tensors.append((name, raw, GGML_TYPE_Q8_0, info['dims']))
                ratio = len(tensor_data) / len(raw) if len(raw) > 0 else 0
                dim_str = "x".join(str(d) for d in reversed(info['dims']))
                print(f"  {name:40s} [{dim_str}] → Q8_0 ({ratio:.1f}x smaller)")

    # Write output GGUF
    print(f"\nWriting {output_path}...")

    # We need to copy the metadata section exactly, but rewrite tensor info + data
    # Simplest: rebuild entire file

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<Q', len(new_tensors)))
        f.write(struct.pack('<Q', len(metadata)))

        # Metadata KV — copy raw bytes from input to avoid array serialization issues
        # Re-read and copy the metadata section byte-by-byte
        with open(input_path, 'rb') as inp:
            inp.seek(24)  # skip magic(4) + version(4) + n_tensors(8) + n_kv(8)
            # We need to re-read metadata to get exact byte range
            for key, vtype, value in metadata:
                # Write key
                write_string(f, key)
                f.write(struct.pack('<I', vtype))
                # Read and copy the value from original
                # Actually simpler to just write it ourselves
                if vtype == 9:  # array
                    # For arrays, re-read from input and copy raw
                    # We need to serialize arrays properly
                    _write_array_value(f, value, key)
                elif vtype == 8:  # string
                    write_string(f, value)
                else:
                    _, fmt, _ = GGUF_TYPES[vtype]
                    f.write(struct.pack(fmt, value))

        # Tensor infos
        # Pre-compute data offsets
        data_offset = 0
        tensor_offsets = []
        for i, (name, raw, ggml_type, dims) in enumerate(new_tensors):
            if i > 0:
                data_offset = ((data_offset + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT
            tensor_offsets.append(data_offset)
            data_offset += len(raw)

        for i, (name, raw, ggml_type, dims) in enumerate(new_tensors):
            write_string(f, name)
            f.write(struct.pack('<I', len(dims)))
            for d in dims:  # dims already in GGML order
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', ggml_type))
            f.write(struct.pack('<Q', tensor_offsets[i]))

        # Alignment padding
        pos = f.tell()
        pad = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        if pad > 0:
            f.write(b'\x00' * pad)

        # Tensor data
        for i, (name, raw, ggml_type, dims) in enumerate(new_tensors):
            # Align
            pos = f.tell()
            pad = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
            if pad > 0:
                f.write(b'\x00' * pad)
            f.write(raw)

    out_size = os.path.getsize(output_path)
    in_size = os.path.getsize(input_path)
    print(f"\nDone!")
    print(f"  Input:  {in_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {out_size / 1024 / 1024:.1f} MB")
    print(f"  Ratio:  {in_size / out_size:.2f}x compression")


def _write_array_value(f, arr, key_hint=""):
    """Write a GGUF array value. Infer element type from content."""
    if not arr:
        # Empty array — default to uint32
        f.write(struct.pack('<I', 4))  # uint32
        f.write(struct.pack('<Q', 0))
        return

    first = arr[0]
    if isinstance(first, str):
        elem_type = 8  # string
    elif isinstance(first, float):
        elem_type = 6  # float32
    elif isinstance(first, int):
        if any(v < 0 for v in arr):
            elem_type = 5  # int32
        else:
            elem_type = 4  # uint32
    elif isinstance(first, list):
        # Nested array — shouldn't happen in practice
        elem_type = 9
    else:
        elem_type = 4  # default uint32

    f.write(struct.pack('<I', elem_type))
    f.write(struct.pack('<Q', len(arr)))

    for elem in arr:
        if elem_type == 8:  # string
            write_string(f, elem)
        elif elem_type == 6:  # float32
            f.write(struct.pack('<f', elem))
        elif elem_type == 5:  # int32
            f.write(struct.pack('<i', elem))
        elif elem_type == 4:  # uint32
            f.write(struct.pack('<I', elem))


if __name__ == "__main__":
    main()
