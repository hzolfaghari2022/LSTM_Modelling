"""Settings for the five-series simulation."""

import os


# 1. Choose each chirp's data role and window settings

# The first three complete chirp experiments are training-only.
TRAIN_ONLY_SHEETS = [
    "DC_Offset_67mA",
    "DC_Offset_87mA",
    "DC_Offset_107mA",
]

# The fourth chirp supplies distributed training, validation, and test
# blocks over its complete time/frequency range.
MIXED_SHEET = "DC_Offset_127mA"

# The fifth chirp is never used for fitting, model selection, scaling,
# or fine-tuning.  It is the pure external test.
FINAL_TEST_SHEET = "DC_Offset_147mA"

# At 500 Hz, 250 samples are 0.5 s.  Repeating this role pattern spreads
# every split over the beginning, middle, and end of the chirp while
# retaining 60% training, 20% validation, and 20% internal test blocks.
BLOCK_SIZE = 250
SPLIT_PATTERN = (
    "training",
    "validation",
    "training",
    "test",
    "training",
)

# Anti-aliased reduction from the original 2000 Hz to 500 Hz.
DOWNSAMPLE_FACTOR = 4

# At 500 Hz, 120 samples contain 0.24 s of physical history.
SEQUENCE_LENGTH = 120

# Overlapping windows retain the transient dynamics without changing
# the model.  Validation and tests evaluate every available target.
TRAIN_STRIDE = 2
VALIDATION_STRIDE = 1
TEST_STRIDE = 1


# 2. Set the shared LSTM layer sizes

INPUT_FEATURES = 3
HIDDEN_1 = 32
HIDDEN_2 = 64
HIDDEN_3 = 64
OUTPUTS = 2
DROPOUT = 0.10


# 3. Set optimizer, loss, validation, and stopping values

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

# PowerShell example for a temporary override:
#     $env:DLSTM_EPOCHS="30"
#     python main.py
EPOCHS = int(os.environ.get("DLSTM_EPOCHS", "20"))

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


# 4. Refit briefly after validation selects the best model

# Validation first chooses the best epoch.  The selected model then uses
# training + validation for a few low-rate updates.  Neither test split
# is used here.
FINAL_FINE_TUNE_EPOCHS = 4
FINAL_FINE_TUNE_LEARNING_RATE = 1e-4
