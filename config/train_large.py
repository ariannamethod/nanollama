"""
Training configuration for nanollama large model.
Multi-GPU training (8×A100).

Parameter count: 3,697,868,800 (~3.7B params)
  embed = 32000 * 3200 = 102,400,000
  unembed = 32000 * 3200 = 102,400,000
  per_layer = 109,158,400
  total = 204,800,000 + 32 * 109,158,400 = 3,697,868,800

Note: Uses same depth=32 as medium, but larger width (3200 vs 2048).
Width scaling is more effective than depth at this scale.
"""

# Model architecture
DEPTH = 32
N_EMBD = 3200
N_HEAD = 32
N_KV_HEAD = 8
SEQUENCE_LEN = 2048

# Exact parameter count: 3,697,868,800 (~3.7B)
PARAM_COUNT = 3_697_868_800

# Training
TOTAL_BATCH_SIZE = 2097152  # 2M tokens per step
DEVICE_BATCH_SIZE = 4
MAX_SEQ_LEN = 2048

# Optimization
EMBEDDING_LR = 0.1
UNEMBEDDING_LR = 0.002
MATRIX_LR = 0.01
WARMUP_STEPS = 1500
NUM_STEPS = 100000

# Data
PERSONALITY_RATIO = 0.20

# Hardware
RECOMMENDED_GPU = "8×A100 80GB"
ESTIMATED_TIME = "~96 hours on 8×A100"
