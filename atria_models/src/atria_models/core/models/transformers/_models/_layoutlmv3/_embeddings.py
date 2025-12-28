from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.layoutlmv3.modeling_layoutlmv3 import *  # noqa

from atria_models.core.models.transformers._configs._encoder_model import (
    EmbeddingsConfig,
)
from atria_models.core.models.transformers._embeddings._token import (
    TokenEmbeddingOutputs,
)
from atria_models.core.models.transformers._models._roberta import (
    RoBertaTokenEmbeddings,
)


class LayoutLMv3EmbeddingsConfig(EmbeddingsConfig):
    max_2d_position_embeddings: int = 1024
    coordinate_size: int = 128
    shape_size: int = 128


@dataclass(frozen=True)
class LayoutLMv3EmbeddingOutputs(TokenEmbeddingOutputs):
    layout_embeddings: torch.Tensor | None = None

    def sum(self) -> torch.Tensor:
        total = self.token_embeddings + self.position_embeddings
        if self.token_type_embeddings is not None:
            total = total + self.token_type_embeddings
        if self.layout_embeddings is not None:
            total = total + self.layout_embeddings
        return total

    def to_ordered_dict(self) -> OrderedDict[str, torch.Tensor | None]:
        return OrderedDict(
            token_embeddings=self.token_embeddings,
            position_embeddings=self.position_embeddings,
            token_type_embeddings=self.token_type_embeddings,
        )


class LayoutLMv3Embeddings(RoBertaTokenEmbeddings):
    def __init__(self, config: LayoutLMv3EmbeddingsConfig):
        super().__init__(config=config)
        self._build_model()

    def _build_embeddings(self):
        super()._build_embeddings()
        self._config: LayoutLMv3EmbeddingsConfig
        self.x_position_embeddings = nn.Embedding(
            self._config.max_2d_position_embeddings, self._config.coordinate_size
        )
        self.y_position_embeddings = nn.Embedding(
            self._config.max_2d_position_embeddings, self._config.coordinate_size
        )
        self.h_position_embeddings = nn.Embedding(
            self._config.max_2d_position_embeddings, self._config.shape_size
        )
        self.w_position_embeddings = nn.Embedding(
            self._config.max_2d_position_embeddings, self._config.shape_size
        )

    def _calculate_spatial_position_embeddings(
        self, layout_ids: torch.Tensor
    ) -> torch.Tensor:
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
            torch.clip(layout_ids[:, :, 3] - layout_ids[:, :, 1], 0, 1023)
        )
        w_position_embeddings = self.w_position_embeddings(
            torch.clip(layout_ids[:, :, 2] - layout_ids[:, :, 0], 0, 1023)
        )

        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
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
        ).float()
        return spatial_position_embeddings

    def forward(  # type: ignore[override]
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        layout_ids: torch.Tensor | None = None,
    ) -> LayoutLMv3EmbeddingOutputs:
        embeddings = super().forward(
            token_ids=token_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        batch_size, seq_len = token_ids.shape[0], token_ids.shape[1]
        if layout_ids is None:
            layout_ids = torch.zeros(
                (batch_size, seq_len, 4), dtype=torch.long, device=token_ids.device
            )
        return LayoutLMv3EmbeddingOutputs(
            token_embeddings=embeddings.token_embeddings,
            position_embeddings=embeddings.position_embeddings,
            token_type_embeddings=embeddings.token_type_embeddings,
            layout_embeddings=self._calculate_spatial_position_embeddings(
                layout_ids=layout_ids
            ),
        )
