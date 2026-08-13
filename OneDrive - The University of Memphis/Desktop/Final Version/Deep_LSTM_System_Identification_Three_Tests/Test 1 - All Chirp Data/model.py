"""Stacked LSTM model for displacement and force prediction."""

import torch
import torch.nn as nn

from config import (
    DROPOUT,
    HIDDEN_1,
    HIDDEN_2,
    HIDDEN_3,
    INPUT_FEATURES,
    OUTPUTS,
)


class DeepLSTMSystemIdentifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(
            INPUT_FEATURES,
            HIDDEN_1,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            HIDDEN_1,
            HIDDEN_2,
            batch_first=True,
        )

        self.lstm3 = nn.LSTM(
            HIDDEN_2,
            HIDDEN_3,
            batch_first=True,
        )

        self.dropout = nn.Dropout(DROPOUT)

        # Outputs: displacement and force.
        self.linear = nn.Linear(
            HIDDEN_3,
            OUTPUTS,
        )

    def forward(self, input_sequence):
        x, _ = self.lstm1(input_sequence)
        x = self.dropout(x)

        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x, _ = self.lstm3(x)
        x = self.dropout(x)

        final_state = x[:, -1, :]

        return self.linear(final_state)
