"""Single source of truth for the cumulative 1-to-13 feature study."""

from one_step_lstm import ONE_STEP_FEATURE_NAMES


def retained_feature_indices(feature_count):
    """Return the first N causal feature columns for incremental ablation."""
    feature_count = int(feature_count)
    if not 1 <= feature_count <= len(ONE_STEP_FEATURE_NAMES):
        raise ValueError("feature_count must be between 1 and 13.")
    return tuple(range(feature_count))


def retained_feature_names(feature_count):
    indices = retained_feature_indices(feature_count)
    return tuple(ONE_STEP_FEATURE_NAMES[index] for index in indices)


def parameter_count(feature_count):
    """Exact trainable count for this project's 2-layer 48-unit LSTM/head."""
    # PyTorch LSTM: first layer changes by 4*hidden_size per input feature.
    # The second LSTM layer and 48->32->1 head are fixed.
    return 30_017 + 192 * int(feature_count)

