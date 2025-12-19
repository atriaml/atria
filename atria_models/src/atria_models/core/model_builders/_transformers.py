"""Transformers model builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.model_builders._base import ModelBuilder

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import AutoModel
    from transformers.configuration_utils import PretrainedConfig
logger = get_logger(__name__)


class TransformersModelBuilder(ModelBuilder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self._model_type is not None, (
            "model_type must be specified for TransformersModelBuilder."
        )
        assert self._model_type in {
            "sequence_classification",
            "token_classification",
            "question_answering",
            "image_classification",
        }, (
            f"Unsupported model_type '{self._model_type}' for AtriaModelBuilder. "
            "Supported types are: 'sequence_classification', 'token_classification', 'question_answering', 'image_classification'."
        )

    def uses_num_labels(self) -> bool:
        if self._model_type in {"sequence_classification", "token_classification"}:
            return True
        return False

    def get_auto_config(self, model_name_or_path: str, **kwargs) -> PretrainedConfig:
        from transformers import AutoConfig

        if self.uses_num_labels:
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
        from torch.nn import Linear

        config = self.get_auto_config(model_name_or_path=model_name_or_path, **kwargs)
        logger.info(
            f"Building model '{model_name_or_path}' with parameters:\n{pretty_repr(config, expand_all=True)}"
        )
        task_cls = self.get_task_class()
        if pretrained:
            model = task_cls.from_pretrained(
                model_name_or_path, config=config, cache_dir=self._cache_dir
            )
        else:
            model = task_cls.from_config(config=config)

        if self._model_type == "image_classification":
            assert "num_labels" in kwargs, (
                "`num_labels` must be provided in model initialization kwargs "
                "for this model type."
            )
            model.classifier = Linear(
                model.classifier.in_features, kwargs["num_labels"]
            )
        return model

    def get_task_class(self) -> type[AutoModel]:
        if self._model_type == "sequence_classification":
            from transformers import AutoModelForSequenceClassification

            return AutoModelForSequenceClassification  # type: ignore
        elif self._model_type == "token_classification":
            from transformers import AutoModelForTokenClassification

            return AutoModelForTokenClassification  # type: ignore
        elif self._model_type == "question_answering":
            from transformers import AutoModelForQuestionAnswering

            return AutoModelForQuestionAnswering  # type: ignore
        elif self._model_type == "image_classification":
            from transformers import AutoModelForImageClassification

            return AutoModelForImageClassification  # type: ignore
        else:
            raise ValueError(
                f"Unsupported model_type '{self._model_type}' for TransformersModelBuilder."
            )
