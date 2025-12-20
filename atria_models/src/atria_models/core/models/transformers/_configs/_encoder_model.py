import enum
from typing import Annotated, Literal, Self, TypeVar

from atria_registry import ModuleConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atria_models.core.models.transformers._configs._common import (
    AttentionConfig,
    CheckpointConfig,
    EmbeddingsConfig,
    LayersConfig,
)


class ClassificationSubTask(str, enum.Enum):
    regression = "regression"
    single_label_classification = "single_label_classification"
    multi_label_classification = "multi_label_classification"


class SequenceClassificationHeadConfig(BaseModel):
    type: Literal["sequence_classification"] = "sequence_classification"
    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)
    num_labels: int = 2
    sub_task: ClassificationSubTask = ClassificationSubTask.single_label_classification


class TokenClassificationHeadConfig(BaseModel):
    type: Literal["token_classification"] = "token_classification"
    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)
    num_labels: int = 2


class QuestionAnsweringHeadConfig(BaseModel):
    type: Literal["question_answering"] = "question_answering"
    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)


HeadConfigType = Annotated[
    SequenceClassificationHeadConfig
    | TokenClassificationHeadConfig
    | QuestionAnsweringHeadConfig,
    Field(discriminator="type"),
]


class TransformersEncoderModelConfig(ModuleConfig):
    """Main configuration that combines all sub-configurations."""

    type: Literal["transformers_encoder"] = "transformers_encoder"
    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)

    # Sub-configurations
    embeddings_config: EmbeddingsConfig = EmbeddingsConfig()
    attention_config: AttentionConfig = AttentionConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    layers_config: LayersConfig = LayersConfig()
    head_config: HeadConfigType | None = None
    output_attentions: bool = False
    output_hidden_states: bool = False
    pretrained: bool = True

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values) -> Self:
        if "hidden_size" in values:
            hidden_size = values["hidden_size"]
            values["embeddings_config"]["hidden_size"] = hidden_size
            values["attention_config"]["hidden_size"] = hidden_size
            values["layers_config"]["hidden_size"] = hidden_size
        return values

    def load_from_hf(self, model_name_or_path: str, **kwargs) -> Self:
        # load bert config from hf
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embeddings_config = EmbeddingsConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            mask_token_id=tokenizer.mask_token_id,
            position_embedding_type="absolute",
        )
        attention_config = AttentionConfig(
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_size=config.hidden_size,
        )
        layers_config = LayersConfig(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            chunk_size_feed_forward=getattr(config, "chunk_size_feed_forward", 0),
            hidden_dropout_prob=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            classifier_dropout=getattr(config, "classifier_dropout", None),
        )
        return self.model_copy(
            update={
                "embeddings_config": embeddings_config,
                "attention_config": attention_config,
                "layers_config": layers_config,
            }
        )


T_TransformersEncoderModelConfig = TypeVar(
    "T_TransformersEncoderModelConfig", bound=TransformersEncoderModelConfig
)
