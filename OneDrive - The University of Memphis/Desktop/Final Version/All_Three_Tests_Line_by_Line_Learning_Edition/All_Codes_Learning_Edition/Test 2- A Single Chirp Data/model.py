# Document the purpose of this module or function; Python keeps this text as its docstring.
"""Stacked LSTM model for displacement and force prediction."""

# Import PyTorch for tensors, the LSTM model, training, and prediction.
import torch
# Import PyTorch for tensors, the LSTM model, training, and prediction.
import torch.nn as nn

# Import selected names from config instead of importing its complete namespace.
from config import (
    # Pass `DROPOUT` as the next value required by the surrounding call or collection.
    DROPOUT,
    # Pass `HIDDEN_1` as the next value required by the surrounding call or collection.
    HIDDEN_1,
    # Pass `HIDDEN_2` as the next value required by the surrounding call or collection.
    HIDDEN_2,
    # Pass `HIDDEN_3` as the next value required by the surrounding call or collection.
    HIDDEN_3,
    # Pass `INPUT_FEATURES` as the next value required by the surrounding call or collection.
    INPUT_FEATURES,
    # Pass `OUTPUTS` as the next value required by the surrounding call or collection.
    OUTPUTS,
# Close the current function call, tuple, or grouped expression.
)


# Define the DeepLSTMSystemIdentifier class used below.
class DeepLSTMSystemIdentifier(nn.Module):
    # Document the purpose of this module or function; Python keeps this text as its docstring.
    """Map one current-history window to displacement and force."""

    # Define the __init__ function; its indented lines form the function body.
    def __init__(self):
        # Call `super`; the following indented continuation lines provide its arguments.
        super().__init__()

        # The three LSTM layers read the sequence with increasing width.
        self.lstm1 = nn.LSTM(
            # Pass `INPUT_FEATURES` as the next value required by the surrounding call or collection.
            INPUT_FEATURES,
            # Pass `HIDDEN_1` as the next value required by the surrounding call or collection.
            HIDDEN_1,
            # Pass `True` as the `batch_first` argument of the surrounding function call.
            batch_first=True,
        # Close the current function call, tuple, or grouped expression.
        )

        # Create the second recurrent layer, which maps 32 hidden values at
        # each time step to the wider 64-value representation.
        self.lstm2 = nn.LSTM(
            # Pass `HIDDEN_1` as the next value required by the surrounding call or collection.
            HIDDEN_1,
            # Pass `HIDDEN_2` as the next value required by the surrounding call or collection.
            HIDDEN_2,
            # Pass `True` as the `batch_first` argument of the surrounding function call.
            batch_first=True,
        # Close the current function call, tuple, or grouped expression.
        )

        # Create the third recurrent layer, which processes the 64-value
        # sequence again before the final prediction is made.
        self.lstm3 = nn.LSTM(
            # Pass `HIDDEN_2` as the next value required by the surrounding call or collection.
            HIDDEN_2,
            # Pass `HIDDEN_3` as the next value required by the surrounding call or collection.
            HIDDEN_3,
            # Pass `True` as the `batch_first` argument of the surrounding function call.
            batch_first=True,
        # Close the current function call, tuple, or grouped expression.
        )

        # Dropout is active during training and disabled during evaluation.
        self.dropout = nn.Dropout(DROPOUT)

        # The final layer returns [displacement, force].
        self.linear = nn.Linear(
            # Pass `HIDDEN_3` as the next value required by the surrounding call or collection.
            HIDDEN_3,
            # Pass `OUTPUTS` as the next value required by the surrounding call or collection.
            OUTPUTS,
        # Close the current function call, tuple, or grouped expression.
        )

    # Define the forward function; its indented lines form the function body.
    def forward(self, input_sequence):
        # x keeps one hidden vector for every time step in the window.
        x, _ = self.lstm1(input_sequence)
        # Apply 10-percent dropout to the first layer's sequence during
        # training; evaluation mode automatically disables it.
        x = self.dropout(x)

        # Send the complete first-layer sequence through the second LSTM.
        # The underscore discards its final hidden/cell-state tuple because
        # the next layer needs the full time sequence instead.
        x, _ = self.lstm2(x)
        # Apply the same configured dropout between the second and third LSTMs.
        x = self.dropout(x)

        # Send the second-layer sequence through the third and final LSTM.
        x, _ = self.lstm3(x)
        # Regularize the third-layer output before selecting its final state.
        x = self.dropout(x)

        # The last time step summarizes the complete input history.
        final_state = x[:, -1, :]

        # Return this value to the code that called the current function.
        return self.linear(final_state)
