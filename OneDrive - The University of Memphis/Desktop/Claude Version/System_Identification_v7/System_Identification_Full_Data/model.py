"""
Configuration conditioned deep LSTM.

The plain stacked LSTM used in the first study receives only current derived
channels, so two records that share the same current but carry a different
load mass are indistinguishable to it. The zero current records make that
failure obvious: their input is identically zero for every load mass, yet
their displacement differs by several millimetres.

This network fixes that in three ways.

  1. Three configuration channels (mass ratio, inverse mass ratio and
     natural frequency ratio) are pulled out of the input sequence and used
     to generate a feature wise scale and shift for each recurrent layer.
     That is a FiLM conditioning layer.
  2. Displacement and force get their own output heads. Force is almost a
     static function of current while displacement is strongly dynamic, so
     forcing both through one shared linear map wastes capacity.
  3. A zero initialised linear bypass connects the last input sample
     directly to the outputs. At initialisation it contributes nothing, so
     training starts from the quasi static baseline and the network only
     has to learn what the baseline misses.
"""

import torch
import torch.nn as nn

from config import (
    CONFIGURATION_FEATURE_INDICES,
    CONFIGURATION_FEATURES,
    DROPOUT,
    FILM_HIDDEN,
    HEAD_HIDDEN,
    HIDDEN_1,
    HIDDEN_2,
    HIDDEN_3,
    INPUT_FEATURES,
    OUTPUTS,
)


class FiLM(nn.Module):
    """Produce a per channel scale and shift from the configuration vector."""

    def __init__(self, configuration_size, feature_size, hidden_size=FILM_HIDDEN):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(configuration_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 * feature_size),
        )
        # Start as the identity transform so the first epochs behave like a
        # plain LSTM and the conditioning is learned gradually.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.feature_size = feature_size

    def forward(self, sequence, configuration):
        parameters = self.network(configuration)
        scale, shift = parameters.chunk(2, dim=-1)
        scale = 1.0 + scale
        return sequence * scale.unsqueeze(1) + shift.unsqueeze(1)


class OutputHead(nn.Module):
    """Small independent head for one physical output."""

    def __init__(self, input_size, hidden_size=HEAD_HIDDEN):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.network(state)


class ConfigurationConditionedLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.configuration_indices = list(CONFIGURATION_FEATURE_INDICES)

        self.lstm1 = nn.LSTM(INPUT_FEATURES, HIDDEN_1, batch_first=True)
        self.lstm2 = nn.LSTM(HIDDEN_1, HIDDEN_2, batch_first=True)
        self.lstm3 = nn.LSTM(HIDDEN_2, HIDDEN_3, batch_first=True)

        self.film1 = FiLM(CONFIGURATION_FEATURES, HIDDEN_1)
        self.film2 = FiLM(CONFIGURATION_FEATURES, HIDDEN_2)
        self.film3 = FiLM(CONFIGURATION_FEATURES, HIDDEN_3)

        self.dropout = nn.Dropout(DROPOUT)

        self.displacement_head = OutputHead(HIDDEN_3)
        self.force_head = OutputHead(HIDDEN_3)

        self.bypass = nn.Linear(INPUT_FEATURES, OUTPUTS)
        nn.init.zeros_(self.bypass.weight)
        nn.init.zeros_(self.bypass.bias)

    def forward(self, input_sequence):
        # Configuration channels are constant along a window, so the value at
        # the last sample fully describes the configuration.
        configuration = input_sequence[:, -1, self.configuration_indices]

        hidden, _ = self.lstm1(input_sequence)
        hidden = self.film1(hidden, configuration)
        hidden = self.dropout(hidden)

        hidden, _ = self.lstm2(hidden)
        hidden = self.film2(hidden, configuration)
        hidden = self.dropout(hidden)

        hidden, _ = self.lstm3(hidden)
        hidden = self.film3(hidden, configuration)
        hidden = self.dropout(hidden)

        final_state = hidden[:, -1, :]

        recurrent_output = torch.cat(
            [
                self.displacement_head(final_state),
                self.force_head(final_state),
            ],
            dim=-1,
        )

        return recurrent_output + self.bypass(input_sequence[:, -1, :])


def describe_model(model):
    """Short printable summary of the signal path."""
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        "Signal path:",
        f"  1 coil current -> {INPUT_FEATURES} features "
        "(excitation, configuration and time channels)",
        f"  -> LSTM {HIDDEN_1} + FiLM -> LSTM {HIDDEN_2} + FiLM "
        f"-> LSTM {HIDDEN_3} + FiLM",
        f"  -> displacement head ({HIDDEN_3} -> {HEAD_HIDDEN} -> 1)",
        f"  -> force head        ({HIDDEN_3} -> {HEAD_HIDDEN} -> 1)",
        f"  -> plus zero initialised linear bypass "
        f"({INPUT_FEATURES} -> {OUTPUTS})",
        f"  -> plus quasi static physical baseline (added outside the network)",
        f"Trainable parameters: {parameters:,}",
    ]
    return "\n".join(lines)
