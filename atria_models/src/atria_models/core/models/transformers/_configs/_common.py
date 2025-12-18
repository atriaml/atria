from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingsConfig(BaseModel):
    """Configuration for embedding layers."""

    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)
    vocab_size: int = 30522
    hidden_size: int = 768
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"


class AttentionConfig(BaseModel):
    """Configuration for attention mechanisms."""

    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1
    hidden_size: int = 768

    @property
    def attention_head_size(self) -> int:
        return int(self.hidden_size / self.num_attention_heads)

    @property
    def all_head_size(self) -> int:
        return self.num_attention_heads * self.attention_head_size

    @model_validator(mode="after")
    def validate_dims(self) -> Self:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )
        return self


class LayersConfig(BaseModel):
    """Configuration for overall model architecture."""

    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)
    num_hidden_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    chunk_size_feed_forward: int = 0
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    classifier_dropout: float | None = None


class CheckpointConfig(BaseModel):
    """Configuration for pretrained checkpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)
    pretrained_checkpoint: str = "hf://bert-base-uncased"
    key_mapping: list[tuple[str, str]] = Field(default_factory=list)
