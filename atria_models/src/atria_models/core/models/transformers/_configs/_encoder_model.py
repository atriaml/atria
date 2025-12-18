import enum
from typing import Self, TypeVar

from atria_registry import ModuleConfig
from pydantic import ConfigDict, model_validator

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


class TransformersEncoderModelConfig(ModuleConfig):
    """Main configuration that combines all sub-configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=False, frozen=True)

    # Sub-configurations
    embeddings_config: EmbeddingsConfig = EmbeddingsConfig()
    attention_config: AttentionConfig = AttentionConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    layers_config: LayersConfig = LayersConfig()
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


class SequenceClassificationModelConfig(TransformersEncoderModelConfig):
    num_labels: int = 2
    sub_task: ClassificationSubTask = ClassificationSubTask.single_label_classification


class TokenClassificationModelConfig(TransformersEncoderModelConfig):
    num_labels: int = 2


class QuestionAnsweringModelConfig(TransformersEncoderModelConfig):
    pass


T_TransformersEncoderModelConfig = TypeVar(
    "T_TransformersEncoderModelConfig", bound=TransformersEncoderModelConfig
)
