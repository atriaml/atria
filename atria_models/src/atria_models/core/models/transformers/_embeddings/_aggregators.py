import torch
from torch import nn


class EmbeddingsAggregator(nn.Module):
    def __init__(self, hidden_size: int, layer_norm_eps: float, dropout_prob: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, embeddings: tuple[torch.Tensor, ...]) -> torch.Tensor:
        agg = None
        for emb in embeddings:
            if emb is not None:
                if agg is None:
                    agg = emb
                else:
                    agg += emb
        agg = self.layer_norm(agg)
        agg = self.dropout(agg)
        return agg
