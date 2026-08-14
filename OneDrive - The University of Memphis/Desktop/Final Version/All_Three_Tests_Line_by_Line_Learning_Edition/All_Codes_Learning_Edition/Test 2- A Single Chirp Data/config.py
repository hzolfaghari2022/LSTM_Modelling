# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Configuration for the one-series deep LSTM simulation."""

# Import operating-system tools for environment variables, CPU counts, and Windows checks.
import os


# 1. Select one chirp and set its distributed data roles

# List the five valid workbook sheets that may be selected below.
AVAILABLE_SHEETS = [
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_67mA",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_87mA",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_107mA",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_127mA",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_147mA",
# Close the current list or index expression.
]

# Change this line or set DLSTM_SINGLE_SHEET temporarily.
SINGLE_SERIES_SHEET = os.environ.get(
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DLSTM_SINGLE_SHEET",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_127mA",
# Close the current function call, tuple, or grouped expression.
)

# At 500 Hz, 250 samples cover 0.5 s and can contain a 120-sample window.
# Repeating short blocks spreads every role across the chirp frequencies.
BLOCK_SIZE = 250
# Three training, one validation, and one test block give a 60/20/20 ratio.
SPLIT_PATTERN = (
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "training",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "validation",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "training",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "test",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "training",
# Close the current function call, tuple, or grouped expression.
)

# Divide the original 2000 Hz rate by four to obtain 500 Hz.
DOWNSAMPLE_FACTOR = 4
# At 500 Hz, 120 samples give the model 0.24 s of input history.
SEQUENCE_LENGTH = 120
# Use every second training target to reduce duplicate windows, but use every
# available validation and test target for dense evaluation.
TRAIN_STRIDE = 2
# Score every possible validation target.
VALIDATION_STRIDE = 1
# Score every possible test target.
TEST_STRIDE = 1


# 2. Set the same LSTM layer sizes used by Test 1

# Three features represent current, current change, and the DC operating point.
INPUT_FEATURES = 3
# Keep the same 32-64-64 hidden widths as Tests 1 and 3 for a fair comparison.
HIDDEN_1 = 32
# Expand the first 32-value representation to 64 values in layer two.
HIDDEN_2 = 64
# Keep 64 values in layer three so it can refine the representation.
HIDDEN_3 = 64
# Predict displacement and force together.
OUTPUTS = 2
# Use mild ten-percent dropout between recurrent layers.
DROPOUT = 0.10


# 3. Set optimizer, loss, validation, and stopping values

# A batch of 128 balances memory use and gradient stability.
BATCH_SIZE = 128
# Adam's common 0.001 starting rate is later controlled by validation.
LEARNING_RATE = 1e-3
# Apply only mild L2 regularization to the recurrent weights.
WEIGHT_DECAY = 1e-6
# The one-series test allows up to 60 epochs because it was originally run as
# a longer demonstration; early stopping still selects the useful endpoint.
EPOCHS = int(os.environ.get("DLSTM_EPOCHS", "60"))

# Stop after 12 non-improving validation epochs.
EARLY_STOPPING_PATIENCE = 12
# Reduce the rate after six non-improving epochs, before stopping at 12.
SCHEDULER_PATIENCE = 6
# Halve the learning rate at a validation plateau.
SCHEDULER_FACTOR = 0.5
# Do not reduce below 0.00001.
MIN_LEARNING_RATE = 1e-5
# Ignore validation changes smaller than 0.00001 as numerical noise.
MINIMUM_IMPROVEMENT = 1e-5

# Equal weights give standardized displacement and force equal basic influence.
DISPLACEMENT_LOSS_WEIGHT = 0.50
# Give standardized force the remaining half of the basic loss.
FORCE_LOSS_WEIGHT = 0.50
# A moderate 0.35 emphasis helps the model learn displacement peaks. It is an
# engineering heuristic and should be validated against other values.
DISPLACEMENT_PEAK_WEIGHT = 0.35

# The arbitrary fixed seed makes repeated runs reproducible.
SEED = 123
# Use up to eight CPU threads and fall back to four if detection fails.
CPU_THREADS = min(8, os.cpu_count() or 4)

# Use only four low-rate epochs after validation selection.
FINAL_FINE_TUNE_EPOCHS = 4
# Fine-tune at one tenth of the initial learning rate.
FINAL_FINE_TUNE_LEARNING_RATE = 1e-4
