"""
Training configuration for nanollama nano model.
Can run on laptop (CPU / MPS / any GPU).

Parameter count formula for Llama 3 (with default vocab_size=32000, head_dim=64):
  embed = vocab_size * n_embd = 32000 * 576 = 18,432,000
  unembed = vocab_size * n_embd = 18,432,000
  attn/layer = n_embd * (n_head + 2*n_kv_head + n_head) * head_dim
             = 576 * (9 + 2*9 + 9) * 64 = 576 * 36 * 64 = 1,327,104
  ffn/layer = 3 * n_embd * ffn_hidden = 3 * 576 * 1536 = 2,654,208
  per_layer = 1,327,104 + 2,654,208 = 3,981,312
  total = 36,864,000 + 13 * 3,981,312 = 88,621,056 (~89M params)
"""

# Model architecture — matches README table
DEPTH = 13
N_EMBD = 576
N_HEAD = 9
N_KV_HEAD = 9   # MHA for nano (kv_heads = heads)
SEQUENCE_LEN = 2048

# Exact parameter count: 88,621,056 (~89M)
PARAM_COUNT = 88_621_056

# Training
TOTAL_BATCH_SIZE = 65536  # 64K tokens per step
DEVICE_BATCH_SIZE = 8
MAX_SEQ_LEN = 1024

# Optimization
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WARMUP_STEPS = 100
NUM_STEPS = 2000

# Data
PERSONALITY_RATIO = 0.20
RECOMMENDED_SAMPLES = 200_000  # ~50M tokens, 5 shards. 2K steps × 64K batch = 128M tokens

# Hardware
RECOMMENDED_GPU = "Any GPU / CPU / MPS"
ESTIMATED_TIME = "~1 hour on A100"
