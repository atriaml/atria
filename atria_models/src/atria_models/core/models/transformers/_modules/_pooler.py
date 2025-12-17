import torch
from torch import nn


class DefaultPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def _get_pooled_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, 0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(self._get_pooled_output(hidden_states)))
