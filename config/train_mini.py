"""
Training configuration for nanollama mini model.
Single GPU training.

Parameter count: 149,815,296 (~150M params)
  embed = 32000 * 768 = 24,576,000
  unembed = 32000 * 768 = 24,576,000
  per_layer = 6,291,456
  total = 49,152,000 + 16 * 6,291,456 = 149,815,296
"""

# Model architecture
DEPTH = 16
N_EMBD = 768
N_HEAD = 12
N_KV_HEAD = 4
SEQUENCE_LEN = 2048

# Exact parameter count: 149,815,296 (~150M)
PARAM_COUNT = 149_815_296

# Training
TOTAL_BATCH_SIZE = 262144  # 256K tokens per step
DEVICE_BATCH_SIZE = 16
MAX_SEQ_LEN = 2048

# Optimization
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WARMUP_STEPS = 300
NUM_STEPS = 10000

# Data
PERSONALITY_RATIO = 0.20

# Hardware
RECOMMENDED_GPU = "A100 40GB"
ESTIMATED_TIME = "~6 hours on A100"
