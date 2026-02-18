"""
Training configuration for nanollama large model (~3B params).
Multi-GPU training (8×A100).
"""

# Model architecture
DEPTH = 32
N_EMBD = 3200
N_HEAD = 32
N_KV_HEAD = 8
SEQUENCE_LEN = 2048

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
