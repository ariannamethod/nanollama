"""
Training configuration for nanollama mini model (~120M params).
Single GPU training.
"""

# Model architecture
DEPTH = 16
N_EMBD = 768
N_HEAD = 12
N_KV_HEAD = 4
SEQUENCE_LEN = 2048

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
