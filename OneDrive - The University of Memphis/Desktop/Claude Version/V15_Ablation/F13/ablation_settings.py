"""Feature selection for this cumulative V15 one-step ablation variant."""

# This value is changed in each delivered folder (13, 12, ..., 1).
SELECTED_FEATURE_COUNT = 13

TOTAL_FEATURE_COUNT = 13
if not 1 <= SELECTED_FEATURE_COUNT <= TOTAL_FEATURE_COUNT:
    raise ValueError("SELECTED_FEATURE_COUNT must be between 1 and 13.")

# The requested study removes features cumulatively from the beginning of
# the original list. For example, the 12-feature version removes feature 1,
# and the 11-feature version removes features 1 and 2.
FIRST_RETAINED_INDEX = TOTAL_FEATURE_COUNT - SELECTED_FEATURE_COUNT
SELECTED_FEATURE_INDICES = tuple(
    range(FIRST_RETAINED_INDEX, TOTAL_FEATURE_COUNT)
)
