"""
Training configuration for nanollama big model.  *** PLANNED ***
Multi-GPU training (8×A100 80GB or 8×H100).

Parameter count: 7,014,973,440 (~7.0B params)
  embed = 96000 * 4096 = 393,216,000
  unembed = 96000 * 4096 = 393,216,000
  per_layer = 173,015,040
  total = 786,432,000 + 36 * 173,015,040 = 7,014,973,440

Architecture: d=4096 (standard for 7-8B class), depth=36.
Vocab: 96K (Tier 3 multilingual — all 13 languages).
Same tier as large but deeper and wider — the flagship model.

Comparable to: Llama 3.2 8B, Mistral 7B, Qwen 2.5 7B.
"""

# Model architecture
DEPTH = 36
N_EMBD = 4096
N_HEAD = 64         # head_dim = 64 (our convention)
N_KV_HEAD = 8
SEQUENCE_LEN = 2048

# Exact parameter count: 7,014,973,440 (~7.0B)
PARAM_COUNT = 7_014_973_440

# Training
TOTAL_BATCH_SIZE = 4194304  # 4M tokens per step
DEVICE_BATCH_SIZE = 4
MAX_SEQ_LEN = 2048

# Optimization (scaled down from large)
EMBEDDING_LR = 0.08
UNEMBEDDING_LR = 0.0015
MATRIX_LR = 0.008
WARMUP_STEPS = 2000
NUM_STEPS = 50000

# Data
PERSONALITY_RATIO = 0.20
RECOMMENDED_SAMPLES = 10_000_000  # For FineWeb-only fallback
# Multi-corpus (default for big): 20B tokens via lambda_train.sh

# Tokenizer
VOCAB_SIZE = 96000  # Tier 3 multilingual (all 13 languages)
TOKENIZER_TIER = 3

# Hardware
RECOMMENDED_GPU = "8×A100 80GB or 8×H100"
ESTIMATED_TIME = "~200 hours on 8×A100, ~80 hours on 8×H100"

# Status
STATUS = "PLANNED"
