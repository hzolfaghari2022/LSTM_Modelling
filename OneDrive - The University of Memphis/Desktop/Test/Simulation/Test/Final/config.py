"""
Project settings for the accuracy-improved, still-simple FARNN model.

The network remains:

    [I, delta_I, I_DC] -> LSTM 32 -> LSTM 64 -> LSTM 64 -> Linear 2

Only targeted training changes were added to improve prediction accuracy.
"""

import os


# ---------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------

# Four complete chirp experiments are used for development.
DEVELOPMENT_SHEETS = [
    "DC_Offset_67mA",
    "DC_Offset_87mA",
    "DC_Offset_107mA",
    "DC_Offset_127mA",
]

# The complete 147 mA experiment remains untouched until the final test.
FINAL_TEST_SHEET = "DC_Offset_147mA"

# Distributed blocks keep low-, medium-, and high-frequency samples in
# both training and validation.
BLOCK_SIZE = 500

# Anti-aliased reduction from 2000 Hz to 500 Hz.
DOWNSAMPLE_FACTOR = 4

# After anti-aliased downsampling, dt = 0.002 s.
# Therefore, 120 samples represent 0.24 s of physical history.
SEQUENCE_LENGTH = 120

# A stride of 5 preserves enough overlapping examples after downsampling.
TRAIN_STRIDE = 5

# Validation is evaluated more densely than training.
VALIDATION_STRIDE = 5

# Predict every available final-test sample.
TEST_STRIDE = 1


# ---------------------------------------------------------------------
# 2. NETWORK
# ---------------------------------------------------------------------

# One measured input is converted into three simple features:
# current, current change, and experiment DC operating point.
INPUT_FEATURES = 3

# Keep the successful simple My Plant architecture.
HIDDEN_1 = 32
HIDDEN_2 = 64
HIDDEN_3 = 64
OUTPUTS = 2

# Lower dropout preserves more dynamic information while still providing
# regularization.
DROPOUT = 0.10


# ---------------------------------------------------------------------
# 3. TRAINING
# ---------------------------------------------------------------------

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

# Train long enough, but stop automatically when validation stops
# improving. A temporary PowerShell override is also supported:
#
#     $env:FARNN_EPOCHS="20"
#     python main.py
EPOCHS = int(
    os.environ.get(
        "FARNN_EPOCHS",
        "20",
    )
)

EARLY_STOPPING_PATIENCE = 12
SCHEDULER_PATIENCE = 6
SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5
MINIMUM_IMPROVEMENT = 1e-5

# Both outputs are standardized, so their losses have comparable scale.
# Displacement receives a little more weight because its resonance
# amplitude was the weakest part of the earlier prediction.
DISPLACEMENT_LOSS_WEIGHT = 0.50
FORCE_LOSS_WEIGHT = 0.50

# Large displacement targets are uncommon but important. This moderate
# weight prevents the long small-amplitude region from dominating MSE.
DISPLACEMENT_PEAK_WEIGHT = 0.35

SEED = 123
CPU_THREADS = min(8, os.cpu_count() or 4)


# ---------------------------------------------------------------------
# 4. FINAL DEVELOPMENT-DATA FINE-TUNING
# ---------------------------------------------------------------------

# After validation selects the best model, use all four development
# records for a few low-learning-rate updates. The 147 mA record remains
# completely untouched.
FINAL_FINE_TUNE_EPOCHS = 4
FINAL_FINE_TUNE_LEARNING_RATE = 1e-4
