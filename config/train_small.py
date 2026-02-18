"""
Training configuration for nanollama small model (~350M params).
Single A100 training.
"""

# Model architecture
DEPTH = 24
N_EMBD = 1024
N_HEAD = 16
N_KV_HEAD = 4
SEQUENCE_LEN = 2048

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

# Hardware
RECOMMENDED_GPU = "A100 80GB"
ESTIMATED_TIME = "~18 hours on A100"
