"""
Training configuration for nanollama goldie model.
First multilingual model in the series (Tier 1: EN, RU, FR, DE).

Parameter count: 1,098,383,360 (~1.1B params)
  embed = 48000 * 2048 = 98,304,000
  unembed = 48000 * 2048 = 98,304,000
  per_layer = 45,088,768
  total = 196,608,000 + 20 * 45,088,768 = 1,098,383,360

Architecture: same dim as medium (2048), fewer layers (20 vs 32).
Vocab: 48K (Tier 1 multilingual tokenizer trained on CulturaX + FineWeb-Edu).
Languages: English, Russian, French, German.

Goldie is the bridge between monolingual small models and multilingual medium/large.
"""

# Model architecture
DEPTH = 20
N_EMBD = 2048
N_HEAD = 32
N_KV_HEAD = 8
SEQUENCE_LEN = 2048

# Exact parameter count: 1,098,383,360 (~1.1B)
PARAM_COUNT = 1_098_383_360

# Training
TOTAL_BATCH_SIZE = 1048576  # 1M tokens per step
DEVICE_BATCH_SIZE = 8
MAX_SEQ_LEN = 2048

# Optimization (interpolated between small and medium)
EMBEDDING_LR = 0.18
UNEMBEDDING_LR = 0.0035
MATRIX_LR = 0.018
WARMUP_STEPS = 700
NUM_STEPS = 30000

# Data
PERSONALITY_RATIO = 0.20
RECOMMENDED_SAMPLES = 5_000_000  # For FineWeb-only fallback (~1.25B tokens)
# Multi-corpus (default for goldie): 3B tokens via lambda_train.sh

# Tokenizer
VOCAB_SIZE = 48000  # Tier 1 multilingual (EN, RU, FR, DE)
TOKENIZER_TIER = 1

# Hardware
RECOMMENDED_GPU = "1-2×A100 80GB"
ESTIMATED_TIME = "~24 hours on 1×A100 80GB"
