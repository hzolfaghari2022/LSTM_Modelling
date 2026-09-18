"""Torch-independent specification shared by all cumulative ablation cases."""

ONE_STEP_FEATURE_NAMES = (
    "current",
    "current_change",
    "current_dc_estimate",
    "mass_ratio",
    "inverse_mass_ratio",
    "elapsed_time",
    "startup_indicator",
    "previous_displacement",
    "previous_velocity",
    "previous_acceleration",
    "previous_force",
    "force_change",
    "quadratic_displacement_baseline",
)


def trainable_parameter_count(feature_count):
    """Parameter count for 2x48 LSTM plus the 48->32->1 dense head."""
    feature_count = int(feature_count)
    if not 1 <= feature_count <= 13:
        raise ValueError("feature_count must be from 1 through 13.")
    hidden = 48
    # PyTorch LSTM has input/recurrent weights and two biases for four gates.
    first_lstm = 4 * hidden * (feature_count + hidden + 2)
    second_lstm = 4 * hidden * (hidden + hidden + 2)
    dense_head = hidden * 32 + 32 + 32 * 1 + 1
    return first_lstm + second_lstm + dense_head
