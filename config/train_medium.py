"""
Training configuration for nanollama medium model.
Multi-GPU training (4×A100).

Parameter count: 1,573,912,576 (~1.6B params)
  embed = 32000 * 2048 = 65,536,000
  unembed = 32000 * 2048 = 65,536,000
  per_layer = 45,088,768
  total = 131,072,000 + 32 * 45,088,768 = 1,573,912,576

Note: Uses same depth=32 as large, but smaller width.
Width scaling is more effective than depth at this scale.
"""

# Model architecture
DEPTH = 32
N_EMBD = 2048
N_HEAD = 32
N_KV_HEAD = 8
SEQUENCE_LEN = 2048

# Exact parameter count: 1,573,912,576 (~1.6B)
PARAM_COUNT = 1_573_912_576

# Training
TOTAL_BATCH_SIZE = 1048576  # 1M tokens per step
DEVICE_BATCH_SIZE = 8
MAX_SEQ_LEN = 2048

# Optimization
EMBEDDING_LR = 0.15
UNEMBEDDING_LR = 0.003
MATRIX_LR = 0.015
WARMUP_STEPS = 1000
NUM_STEPS = 50000

# Data
PERSONALITY_RATIO = 0.20
RECOMMENDED_SAMPLES = 10_000_000  # ~2.5B tokens, 250 shards. 50K steps × 1M batch = 50B tokens

# Hardware
RECOMMENDED_GPU = "4×A100 80GB"
ESTIMATED_TIME = "~48 hours on 4×A100"
