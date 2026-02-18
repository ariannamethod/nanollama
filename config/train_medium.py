"""
Training configuration for nanollama medium model (~1B params).
Multi-GPU training (4×A100).
"""

# Model architecture
DEPTH = 32
N_EMBD = 2048
N_HEAD = 32
N_KV_HEAD = 8
SEQUENCE_LEN = 2048

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

# Hardware
RECOMMENDED_GPU = "4×A100 80GB"
ESTIMATED_TIME = "~48 hours on 4×A100"
