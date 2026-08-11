"""Configuration for the one-series FARNN Python simulation."""

import os


# ---------------------------------------------------------------------
# 1. SELECT ONE OF THE FIVE CHIRP RECORDS
# ---------------------------------------------------------------------

AVAILABLE_SHEETS = [
    "DC_Offset_67mA",
    "DC_Offset_87mA",
    "DC_Offset_107mA",
    "DC_Offset_127mA",
    "DC_Offset_147mA",
]

# Change this one line, or set FARNN_SINGLE_SHEET temporarily.
SINGLE_SERIES_SHEET = os.environ.get(
    "FARNN_SINGLE_SHEET",
    "DC_Offset_127mA",
)

# At 500 Hz, each 250-sample block covers 0.5 s.  This repeating pattern
# spreads all three roles across low, medium, and high chirp frequencies.
BLOCK_SIZE = 250
SPLIT_PATTERN = (
    "training",
    "validation",
    "training",
    "test",
    "training",
)

DOWNSAMPLE_FACTOR = 4
SEQUENCE_LENGTH = 120
TRAIN_STRIDE = 2
VALIDATION_STRIDE = 1
TEST_STRIDE = 1


# ---------------------------------------------------------------------
# 2. PAPER-ALIGNED, PLANT-SPECIFIC NETWORK
# ---------------------------------------------------------------------

INPUT_FEATURES = 3
HIDDEN_1 = 32
HIDDEN_2 = 64
HIDDEN_3 = 64
OUTPUTS = 2
DROPOUT = 0.10


# ---------------------------------------------------------------------
# 3. TRAINING
# ---------------------------------------------------------------------

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
# The verified default reaches the validation minimum before early
# stopping (best epoch 38 for the included 127 mA record).
EPOCHS = int(os.environ.get("FARNN_EPOCHS", "60"))

EARLY_STOPPING_PATIENCE = 12
SCHEDULER_PATIENCE = 6
SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5
MINIMUM_IMPROVEMENT = 1e-5

DISPLACEMENT_LOSS_WEIGHT = 0.50
FORCE_LOSS_WEIGHT = 0.50
DISPLACEMENT_PEAK_WEIGHT = 0.35

SEED = 123
CPU_THREADS = min(8, os.cpu_count() or 4)

FINAL_FINE_TUNE_EPOCHS = 4
FINAL_FINE_TUNE_LEARNING_RATE = 1e-4
