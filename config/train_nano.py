"""
Training configuration for nanollama nano model (~15M params).
Smoke test, can run on laptop.
"""

# Model architecture
DEPTH = 6
N_EMBD = 384
N_HEAD = 6
N_KV_HEAD = 2
SEQUENCE_LEN = 2048

# Training
TOTAL_BATCH_SIZE = 65536  # 64K tokens per step
DEVICE_BATCH_SIZE = 8
MAX_SEQ_LEN = 1024  # Shorter for quick tests

# Optimization
EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.02
WARMUP_STEPS = 100
NUM_STEPS = 1000

# Data
PERSONALITY_RATIO = 0.20

# Hardware
RECOMMENDED_GPU = "Any GPU / CPU / MPS"
ESTIMATED_TIME = "~30 minutes on A100"
