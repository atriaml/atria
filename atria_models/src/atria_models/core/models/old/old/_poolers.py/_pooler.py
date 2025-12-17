import torch
from torch import nn


class DefaultPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def _get_pooled_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, 0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(self._get_pooled_output(hidden_states)))
