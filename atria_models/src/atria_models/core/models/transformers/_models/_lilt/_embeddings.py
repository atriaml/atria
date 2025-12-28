from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from atria_models.core.models.transformers._configs._encoder_model import (
    EmbeddingsConfig,
)
from atria_models.core.models.transformers._models._roberta import (
    RoBertaTokenEmbeddings,
)


class LiLTEmbeddingsConfig(EmbeddingsConfig):
    max_2d_position_embeddings: int = 1024
    channel_shrink_ratio: int = 4


@dataclass(frozen=True)
class LiLTEmbeddingOutputs:
    token_embeddings: torch.Tensor
    position_embeddings: torch.Tensor | None = None
    token_type_embeddings: torch.Tensor | None = None
    layout_embeddings: torch.Tensor | None = None

    def to_ordered_dict(self) -> OrderedDict[str, torch.Tensor | None]:
        return OrderedDict(
            token_embeddings=self.token_embeddings,
            position_embeddings=self.position_embeddings,
            token_type_embeddings=self.token_type_embeddings,
        )


class LayoutEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int,
        max_2d_position_embeddings: int,
        pad_token_id: int,
        channel_shrink_ratio: int,
    ):
        super().__init__()
        # we divide the hidden_size by 6 here as there are 6 different layout embeddings,
        # namely left_position, upper_position, right_position, lower_position, height, width
        self.x_position_embeddings = nn.Embedding(
            max_2d_position_embeddings, hidden_size // 6
        )
        self.y_position_embeddings = nn.Embedding(
            max_2d_position_embeddings, hidden_size // 6
        )
        self.h_position_embeddings = nn.Embedding(
            max_2d_position_embeddings, hidden_size // 6
        )
        self.w_position_embeddings = nn.Embedding(
            max_2d_position_embeddings, hidden_size // 6
        )

        self.padding_idx = pad_token_id
        self.box_position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size // channel_shrink_ratio,
            padding_idx=self.padding_idx,
        )
        self.box_linear_embeddings = nn.Linear(
            in_features=hidden_size, out_features=hidden_size // channel_shrink_ratio
        )

    def forward(self, layout_ids: torch.Tensor, position_ids: torch.Tensor):
        try:
            left_position_embeddings = self.x_position_embeddings(layout_ids[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(layout_ids[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(layout_ids[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(layout_ids[:, :, 3])
        except IndexError as e:
            raise IndexError(
                "The `bbox` coordinate values should be within 0-1000 range."
            ) from e

        h_position_embeddings = self.h_position_embeddings(
            layout_ids[:, :, 3] - layout_ids[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            layout_ids[:, :, 2] - layout_ids[:, :, 0]
        )
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        box_position_embeddings = self.box_position_embeddings(position_ids)
        spatial_position_embeddings = self.box_linear_embeddings(
            spatial_position_embeddings
        )
        spatial_position_embeddings = (
            spatial_position_embeddings + box_position_embeddings
        )
        return spatial_position_embeddings


class LiLTEmbeddings(RoBertaTokenEmbeddings):
    def __init__(self, config: LiLTEmbeddingsConfig):
        super().__init__(config=config)
        self._build_model()

    def _build_embeddings(self):
        super()._build_embeddings()
        self._config: LiLTEmbeddingsConfig
        self.layout_embeddings = LayoutEmbeddings(
            hidden_size=self._config.hidden_size,
            max_position_embeddings=self._config.max_position_embeddings,
            max_2d_position_embeddings=self._config.max_2d_position_embeddings,
            pad_token_id=self._config.pad_token_id,
            channel_shrink_ratio=self._config.channel_shrink_ratio,
        )

    def forward(  # type: ignore[override]
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        layout_ids: torch.Tensor | None = None,
        past_kv_length: int = 0,
    ) -> LiLTEmbeddingOutputs:
        # input shape
        batch_size, seq_length = token_ids.size()

        # default layout ids
        if layout_ids is None:
            layout_ids = torch.zeros(
                (batch_size, seq_length) + (4,),
                dtype=torch.long,
                device=token_ids.device,
            )

        # resolve ids
        if position_ids is None:
            position_ids = self._default_position_ids(token_ids, past_kv_length)
        if token_type_ids is None:
            token_type_ids = self._default_token_type_ids(
                position_ids, batch_size, seq_length
            )

        # embeddings
        return LiLTEmbeddingOutputs(
            token_embeddings=self.word_embeddings(token_ids),
            position_embeddings=self.position_embeddings(position_ids),
            token_type_embeddings=self.token_type_embeddings(token_type_ids),
            layout_embeddings=self.layout_embeddings(
                layout_ids=layout_ids, position_ids=position_ids
            ),
        )


class LiLTEmbeddingsPostProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_norm_eps: float,
        dropout_prob: float,
        channel_shrink_ratio: int,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
        self.layout_layer_norm = nn.LayerNorm(
            hidden_size // channel_shrink_ratio, eps=layer_norm_eps
        )
        self.layout_dropout = nn.Dropout(dropout_prob)

    def forward(
        self, embeddings: LiLTEmbeddingOutputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # post-process layout embeddings
        layout_embeddings = self.layout_layer_norm(embeddings.layout_embeddings)
        layout_embeddings = self.dropout(layout_embeddings)

        # post-process other embeddings
        token_embeddings = embeddings.token_embeddings + embeddings.position_embeddings
        if embeddings.token_type_embeddings is not None:
            token_embeddings = token_embeddings + embeddings.token_type_embeddings
        token_embeddings = self.layer_norm(token_embeddings)
        token_embeddings = self.dropout(token_embeddings)
        return token_embeddings, layout_embeddings
