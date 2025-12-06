from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.data_types._image import ImageTensorDataModel
from atria_transforms.tfs._image_processor._base import ImageProcessor
from atria_types._common import TrainingStage
from pydantic import BaseModel, model_validator

from atria_models.core.model_pipelines._common import ModelPipelineConfig
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_models.core.types.model_outputs import ClassificationModelOutput, ModelOutput
from atria_models.registry import MODEL_PIPELINE

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class MixupConfig(BaseModel):
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 0.0
    cutmix_minmax: float | None = None
    prob: float = 1.0
    switch_prob: float = 0.5
    mode: str = "batch"
    mixup_prob: float = 1.0
    correct_lam: bool = True
    label_smoothing: float = 0.1


class ImageModelPipelineConfig(ModelPipelineConfig):
    mixup_config: MixupConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_transforms(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "train_transform" not in values or values["train_transform"] is None:
            values["train_transform"] = ImageProcessor()
        else:
            values["train_transform"] = ImageProcessor.model_validate(
                values["train_transform"]
            )
        if "eval_transform" not in values or values["eval_transform"] is None:
            values["eval_transform"] = ImageProcessor()
        else:
            values["eval_transform"] = ImageProcessor.model_validate(
                values["eval_transform"]
            )
        return values


class ImageModelPipeline(ModelPipeline[ImageModelPipelineConfig]):
    __config__ = ImageModelPipelineConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from timm.data.mixup import Mixup
        from timm.loss.cross_entropy import (
            LabelSmoothingCrossEntropy,
            SoftTargetCrossEntropy,
        )
        from torch import nn

        assert self._labels.classification is not None, (
            "Labels must be provided for classification tasks."
        )
        if self.config.mixup_config is not None:
            self._mixup = Mixup(
                num_classes=len(self._labels.classification),
                mixup_alpha=self.config.mixup_config.mixup_alpha,
                cutmix_alpha=self.config.mixup_config.cutmix_alpha,
                cutmix_minmax=self.config.mixup_config.cutmix_minmax,
                label_smoothing=self.config.mixup_config.label_smoothing,
            )
            self._loss_fn_train = (
                LabelSmoothingCrossEntropy(self._mixup.label_smoothing)
                if self._mixup.label_smoothing > 0.0
                else SoftTargetCrossEntropy()
            )
        else:
            self._loss_fn_train = nn.CrossEntropyLoss()
        self._loss_fn_eval = nn.CrossEntropyLoss()

    def training_step(  # type: ignore[override]
        self, batch: ImageTensorDataModel | DocumentTensorDataModel, **kwargs
    ) -> ModelOutput:
        assert batch.label is not None, "Labels cannot be None"
        inputs = self._input_transform(batch=batch)
        model_output = self._model(**inputs)
        logits = (
            model_output.logits if hasattr(model_output, "logits") else model_output
        )
        loss = self._loss_fn_train(logits, batch.label)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def evaluation_step(  # type: ignore[override]
        self,
        batch: ImageTensorDataModel | DocumentTensorDataModel,
        stage: TrainingStage,
        **kwargs,
    ) -> ModelOutput:
        assert batch.label is not None, "Labels cannot be None"
        inputs = self._input_transform(batch)
        model_output = self._model(**inputs)
        logits = (
            model_output.logits if hasattr(model_output, "logits") else model_output
        )
        loss = self._loss_fn_eval(logits, batch.label)
        return self._output_transform(loss=loss, model_output=model_output, batch=batch)

    def predict_step(  # type: ignore[override]
        self, batch: ImageTensorDataModel | DocumentTensorDataModel, **kwargs
    ) -> ModelOutput:
        inputs = self._input_transform(batch)
        model_output = self._model(**inputs)
        return self._output_transform(loss=None, model_output=model_output, batch=batch)

    def _model_build_kwargs(self) -> dict[str, object]:
        assert self._labels.classification is not None, (
            f"'classification' labels must be provided for the {self.__class__.__name__} pipeline. Found: {self._labels}"
        )
        return {"num_labels": len(self._labels.classification)}

    def _input_transform(
        self, batch: ImageTensorDataModel | DocumentTensorDataModel
    ) -> dict[str, torch.Tensor | None]:
        assert batch.image is not None, "Image cannot be None"
        if "pixel_values" in self._model_args_list:
            return {"pixel_values": batch.image}
        elif len(self._model_args_list) == 1:
            return {self._model_args_list[0]: batch.image}
        else:
            raise ValueError(
                f"Model forward method has unsupported arguments: {self._model_args_list}"
            )

    def _output_transform(
        self,
        loss: torch.Tensor | None,
        model_output: Any,
        batch: ImageTensorDataModel | DocumentTensorDataModel,
    ) -> ModelOutput:
        assert self._labels.classification is not None, (
            "Labels must be provided for classification tasks."
        )
        assert batch.label is not None, "Labels cannot be None"
        logits = (
            model_output.logits if hasattr(model_output, "logits") else model_output
        )
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
            prediction_probs=logits.softmax(dim=-1),
            gt_label_value=batch.label if batch.label is not None else None,
            gt_label_name=[self._labels.classification[i] for i in batch.label.tolist()]
            if batch.label is not None
            else None,
            predicted_label_value=predicted_labels
            if predicted_labels is not None
            else None,
            predicted_label_name=[
                self._labels.classification[i] for i in predicted_labels.tolist()
            ]
            if predicted_labels is not None
            else None,
        )


@MODEL_PIPELINE.register("image_classification")
class ImageClassificationPipeline(ImageModelPipeline):
    __config__ = ImageModelPipelineConfig
