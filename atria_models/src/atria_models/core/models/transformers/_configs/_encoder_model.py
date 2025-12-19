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


T_TransformersEncoderModelConfig = TypeVar(
    "T_TransformersEncoderModelConfig", bound=TransformersEncoderModelConfig
)
