"""Transformers model builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.model_builders._base import ModelBuilder
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import AutoModel
    from transformers.configuration_utils import PretrainedConfig
logger = get_logger(__name__)


class TransformersModelBuilder(ModelBuilder):
    __uses_num_labels__: bool = True

    def get_auto_config(self, model_name_or_path: str, **kwargs) -> PretrainedConfig:
        from transformers import AutoConfig

        if self.__uses_num_labels__:
            assert "num_labels" in kwargs, (
                "`num_labels` must be provided in model initialization kwargs "
                "for this model type."
            )

        return AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=self._cache_dir, **kwargs
        )

    def _build(
        self, model_name_or_path: str, pretrained: bool = True, **kwargs
    ) -> Module:
        config = self.get_auto_config(model_name_or_path=model_name_or_path, **kwargs)
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


@MODEL_PIPELINE.register("transformers/sequence_classification")
class SequenceClassificationModelBuilder(TransformersModelBuilder):
    __uses_num_labels__: bool = True

    def get_task_class(self) -> type[AutoModel]:
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification  # type: ignore


@MODEL_PIPELINE.register("transformers/token_classification")
class TokenClassificationModelBuilder(TransformersModelBuilder):
    __uses_num_labels__: bool = True

    def get_task_class(self) -> type[AutoModel]:
        from transformers import AutoModelForTokenClassification

        return AutoModelForTokenClassification  # type: ignore


@MODEL_PIPELINE.register("transformers/question_answering")
class QuestionAnsweringModelBuilder(TransformersModelBuilder):
    __uses_num_labels__: bool = False

    def get_task_class(self) -> type[AutoModel]:
        from transformers import AutoModelForQuestionAnswering

        return AutoModelForQuestionAnswering  # type: ignore


@MODEL_PIPELINE.register("transformers/image_classification")
class ImageClassificationModelBuilder(TransformersModelBuilder):
    __uses_num_labels__: bool = False

    def get_task_class(self) -> type[AutoModel]:
        from transformers import AutoModelForImageClassification

        return AutoModelForImageClassification  # type: ignore

    def _build(
        self, model_name_or_path: str, pretrained: bool = True, **kwargs
    ) -> Module:
        from torch.nn import Linear

        model = super()._build(
            model_name_or_path=model_name_or_path, pretrained=pretrained, **kwargs
        )
        assert "num_labels" in kwargs, (
            "`num_labels` must be provided in model initialization kwargs "
            "for this model type."
        )
        model.classifier = Linear(model.classifier.in_features, kwargs["num_labels"])
        return model
