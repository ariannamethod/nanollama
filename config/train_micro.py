"""
Training configuration for nanollama micro model (~50M params).
Single GPU training.
"""

# Model architecture
DEPTH = 12
N_EMBD = 512
N_HEAD = 8
N_KV_HEAD = 2
SEQUENCE_LEN = 2048

# Training
TOTAL_BATCH_SIZE = 131072  # 128K tokens per step
DEVICE_BATCH_SIZE = 16
MAX_SEQ_LEN = 2048

# Optimization
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WARMUP_STEPS = 200
NUM_STEPS = 5000

# Data
PERSONALITY_RATIO = 0.20

# Hardware
RECOMMENDED_GPU = "RTX 3090 / A10 / A100"
ESTIMATED_TIME = "~2 hours on A100"
