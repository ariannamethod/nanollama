"""
Training configuration for nanollama small model.
Single A100 training.

Parameter count: 336,068,608 (~336M params)
  embed = 32000 * 1024 = 32,768,000
  unembed = 32000 * 1024 = 32,768,000
  per_layer = 11,272,192
  total = 65,536,000 + 24 * 11,272,192 = 336,068,608
"""

# Model architecture
DEPTH = 24
N_EMBD = 1024
N_HEAD = 16
N_KV_HEAD = 4
SEQUENCE_LEN = 2048

# Exact parameter count: 336,068,608 (~336M)
PARAM_COUNT = 336_068_608

# Training
TOTAL_BATCH_SIZE = 524288  # 512K tokens per step
DEVICE_BATCH_SIZE = 16
MAX_SEQ_LEN = 2048

# Optimization
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WARMUP_STEPS = 500
NUM_STEPS = 20000

# Data
PERSONALITY_RATIO = 0.20
RECOMMENDED_SAMPLES = 3_000_000  # ~750M tokens, 75 shards. 20K steps × 512K batch = 10.5B tokens

# Hardware
RECOMMENDED_GPU = "A100 80GB"
ESTIMATED_TIME = "~18 hours on 1×A100"
