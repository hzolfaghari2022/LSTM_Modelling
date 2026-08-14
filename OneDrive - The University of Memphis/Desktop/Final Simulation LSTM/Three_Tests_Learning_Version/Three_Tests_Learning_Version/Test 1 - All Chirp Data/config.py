# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Settings for the five-series simulation."""

# Import operating-system tools for environment variables, CPU counts, and Windows checks.
import os


# 1. Choose each chirp's data role and window settings

# The first three complete chirp experiments are training-only.
TRAIN_ONLY_SHEETS = [
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_67mA",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_87mA",
    # Supply this exact text value to the surrounding call, list, tuple, or dictionary.
    "DC_Offset_107mA",
# Close the current list or index expression.
]

# The fourth chirp supplies distributed training, validation, and test
# blocks over its complete time/frequency range.
MIXED_SHEET = "DC_Offset_127mA"

# The fifth chirp is never used for fitting, model selection, scaling,
# or fine-tuning.  It is the pure external test.
FINAL_TEST_SHEET = "DC_Offset_147mA"

# At 500 Hz, 250 samples are 0.5 s. This is long enough to contain the
# 120-sample history while still creating several blocks across the chirp.
BLOCK_SIZE = 250
# Three of every five blocks are training, one is validation, and one is
# test. This gives a 60/20/20 role ratio spread over the frequency range.
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

# The original chirp is sampled 2,000 times per second. Keeping one quarter
# of that information gives 2,000 / 4 = 500 samples per second. A 31 Hz wave
# needs at least 62 samples per second by the Nyquist rule, so 500 Hz still
# represents it safely. data_utils.py uses resample_poly so high-frequency
# content is filtered before samples are removed.
DOWNSAMPLE_FACTOR = 4

# At 500 Hz, 120 samples contain 0.24 s of history. This is a modeling
# choice that captures short transients without making every window large.
SEQUENCE_LENGTH = 120

# The stride is how far the sliding window moves before making the next
# example. A 120-sample window starting at sample 0 is followed by one
# starting at sample 2, so the two windows overlap by 118 samples. Using 2
# halves the number of nearly duplicate training examples and reduces time.
TRAIN_STRIDE = 2
# Move one sample at a time in validation so every eligible target is scored.
VALIDATION_STRIDE = 1
# Move one sample at a time in test so the final evaluation is dense.
TEST_STRIDE = 1


# 2. Set the shared LSTM layer sizes

# Three features are supplied at every time step: I, delta-I, and mean I.
INPUT_FEATURES = 3
# The first 32-unit layer makes the initial sequence representation compact.
HIDDEN_1 = 32
# The next two 64-unit layers provide more capacity for nonlinear dynamics.
# These sizes are plant-specific engineering choices, not universal values.
HIDDEN_2 = 64
# Keep 64 units in the third layer so it can refine, rather than compress,
# the representation produced by the second 64-unit layer.
HIDDEN_3 = 64
# Two outputs are required: displacement and Lorentz force.
OUTPUTS = 2
# Ten-percent dropout is mild regularization between LSTM layers. It was
# kept small because the network is not very large and underfitting is a risk.
DROPOUT = 0.10


# 3. Set optimizer, loss, validation, and stopping values

# A batch of 128 balances memory use and stable gradient estimates on CPU.
BATCH_SIZE = 128
# Adam commonly starts at 0.001; validation and the scheduler control it later.
LEARNING_RATE = 1e-3
# This very small weight decay adds mild L2 regularization without strongly
# shrinking the recurrent weights.
WEIGHT_DECAY = 1e-6

# PowerShell example for a temporary override:
#     $env:DLSTM_EPOCHS="30"
#     python main.py
# Twenty is the default comparison budget for Tests 1 and 3. The environment
# variable allows a longer experiment without editing the file. Early stopping
# can finish before 20 when validation no longer improves.
EPOCHS = int(os.environ.get("DLSTM_EPOCHS", "20"))

# Wait through 12 non-improving validation epochs before stopping. This allows
# normal validation fluctuations but avoids continuing indefinitely.
EARLY_STOPPING_PATIENCE = 12
# After six non-improving epochs, lower the learning rate. Six is half of the
# stopping patience, so the model gets time to improve at the reduced rate.
SCHEDULER_PATIENCE = 6
# Multiply the learning rate by 0.5, which means halve it at each plateau.
SCHEDULER_FACTOR = 0.5
# Do not reduce below 0.00001; smaller updates would add runtime with little
# practical parameter movement in this model.
MIN_LEARNING_RATE = 1e-5
# Treat changes smaller than 0.00001 as numerical noise rather than improvement.
MINIMUM_IMPROVEMENT = 1e-5

# Standardization puts displacement and force on comparable scales, so equal
# 0.50 weights make each output contribute half of the basic loss.
DISPLACEMENT_LOSS_WEIGHT = 0.50
# Give standardized force the remaining half of the basic loss.
FORCE_LOSS_WEIGHT = 0.50
# Add a moderate 0.35 emphasis to large displacement targets. This was an
# accuracy-oriented heuristic, not a value specified by the paper; it should
# be checked by comparing values such as 0, 0.20, 0.35, and 0.50 on validation.
DISPLACEMENT_PEAK_WEIGHT = 0.35

# Seed 123 is an arbitrary fixed seed used only for reproducibility. It is not
# selected because it gives higher accuracy, so conclusions should be checked
# across several seeds when reporting statistical robustness.
SEED = 123
# Use the available logical CPU count but cap it at eight to reduce thread
# oversubscription. If Python cannot detect the count, use four as a fallback.
CPU_THREADS = min(8, os.cpu_count() or 4)


# 4. Refit briefly after validation selects the best model

# Four extra epochs are intentionally short so the validation-selected model
# is only adjusted, not completely refitted. Test data remain excluded.
FINAL_FINE_TUNE_EPOCHS = 4
# Use 0.0001, ten times smaller than the initial 0.001 learning rate, to reduce
# the chance of destroying the validation-selected solution.
FINAL_FINE_TUNE_LEARNING_RATE = 1e-4
