"""Transformers model builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from rich.pretty import pretty_repr

from atria_models.api.models import load_model_config
from atria_models.core.model_builders._base import ModelBuilder
from atria_models.core.models.transformers._configs._encoder_model import (
    ClassificationSubTask,
    QuestionAnsweringHeadConfig,
    SequenceClassificationHeadConfig,
    TokenClassificationHeadConfig,
    TransformersEncoderModelConfig,
)

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import AutoModel
logger = get_logger(__name__)


class AtriaModelBuilder(ModelBuilder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self._model_type is not None, (
            "model_type must be specified for TransformersModelBuilder."
        )
        assert self._model_type in {
            "sequence_classification",
            "token_classification",
            "question_answering",
        }, (
            f"Unsupported model_type '{self._model_type}' for AtriaModelBuilder. "
            "Supported types are: 'sequence_classification', 'token_classification', 'question_answering'."
        )

    def get_config(
        self, model_name_or_path: str, **kwargs
    ) -> TransformersEncoderModelConfig:
        config = load_model_config(
            model_name=model_name_or_path, cache_dir=self._cache_dir, **kwargs
        )
        if self._model_type == "sequence_classification":
            head_config = SequenceClassificationHeadConfig(
                num_labels=kwargs.get("num_labels", 2),
                sub_task=kwargs.get(
                    "sub_task", ClassificationSubTask.single_label_classification
                ),
            )
            config = config.model_copy(update={"head_config": head_config})
        elif self._model_type == "token_classification":
            head_config = TokenClassificationHeadConfig(
                num_labels=kwargs.get("num_labels", 2)
            )
            config = config.model_copy(update={"head_config": head_config})
        elif self._model_type == "question_answering":
            head_config = QuestionAnsweringHeadConfig()
            config = config.model_copy(update={"head_config": head_config})
        return config.model_validate(config)

    def _build(
        self, model_name_or_path: str, pretrained: bool = True, **kwargs
    ) -> Module:
        config = self.get_config(model_name_or_path=model_name_or_path, **kwargs)
        logger.info(
            f"Building model '{model_name_or_path}' with parameters:\n{pretty_repr(config, expand_all=True)}"
        )
        task_cls = self.get_task_class()
        if pretrained:
            return task_cls.from_pretrained(
                model_name_or_path, config=config, cache_dir=self._cache_dir
            )
        else:
            return task_cls.from_config(config=config)

    def get_task_class(self) -> type[AutoModel]:
        raise NotImplementedError(
            "Subclasses must implement the `get_task_class` method to return the appropriate model class."
        )
