"""
Training configuration for nanollama nano model.
Smoke test, can run on laptop.

Parameter count formula for Llama 3 (with default vocab_size=32000):
  embed = vocab_size * n_embd (input embeddings)
  unembed = vocab_size * n_embd (output projection, untied)
  attn/layer = n_embd * (n_head + 2*n_kv_head + n_head) * head_dim
  ffn/layer = 3 * n_embd * ffn_hidden
  total = embed + unembed + n_layer * (attn + ffn)

This config (with vocab_size=32000):
  embed = 32000 * 384 = 12,288,000
  unembed = 32000 * 384 = 12,288,000  
  attn/layer = 384 * (6 + 2*2 + 6) * 64 = 393,216
  ffn/layer = 3 * 384 * 1024 = 1,179,648
  total = 24,576,000 + 6 * 1,572,864 = 34,013,184 (~34M params)
"""

# Model architecture
DEPTH = 6
N_EMBD = 384
N_HEAD = 6
N_KV_HEAD = 2
SEQUENCE_LEN = 2048

# Exact parameter count: 34,013,184 (~34M)
PARAM_COUNT = 34_013_184

# Training
TOTAL_BATCH_SIZE = 65536  # 64K tokens per step
DEVICE_BATCH_SIZE = 8
MAX_SEQ_LEN = 1024  # Shorter for quick tests

# Optimization
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WARMUP_STEPS = 100
NUM_STEPS = 1000

# Data
PERSONALITY_RATIO = 0.20
RECOMMENDED_SAMPLES = 200_000  # ~50M tokens, 5 shards. Enough for 1K steps

# Hardware
RECOMMENDED_GPU = "Any GPU / CPU / MPS"
ESTIMATED_TIME = "~30 minutes on A100"
