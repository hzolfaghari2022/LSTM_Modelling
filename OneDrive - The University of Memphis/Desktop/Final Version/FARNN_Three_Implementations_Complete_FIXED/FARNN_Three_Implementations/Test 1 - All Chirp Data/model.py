"""
Ogunmolu-based stacked LSTM adapted to your system.

The paper is the base:
    stacked LSTMs + dropout + final Linear layer

Necessary problem-specific adaptation:
    [I, ΔI, I_DC] -> 32 -> 64 -> 64 -> 2
"""

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


class OgunmoluCOMSOLLSTM(nn.Module):
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

        # Two outputs:
        # output 0 = displacement
        # output 1 = force
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
