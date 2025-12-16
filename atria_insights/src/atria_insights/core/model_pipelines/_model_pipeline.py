from __future__ import annotations

import inspect
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Generic

import torch
from atria_logger import get_logger
from atria_registry._module_base import ConfigurableModule
from atria_transforms.core._data_types._base import T_TensorDataModel
from atria_types._datasets import DatasetLabels
from ignite.metrics import Metric
from torchxai.data_types import (
    ExplanationInputs,
    ExplanationState,
    ExplanationTargetType,
)
from tqdm import tqdm

from atria_insights.core.model_pipelines._common import (
    ExplanationTargetStrategy,
    T_ExplainableModelPipelineConfig,
)

logger = get_logger(__name__)


class ExplainableModelPipeline(
    ConfigurableModule[T_ExplainableModelPipelineConfig],
    Generic[T_ExplainableModelPipelineConfig, T_TensorDataModel],
):
    __abstract__ = True
    __config__: type[T_ExplainableModelPipelineConfig]

    def __init__(
        self, config: T_ExplainableModelPipelineConfig, labels: DatasetLabels
    ) -> None:
        super().__init__(config=config)

        self._model_pipeline = self.config.model_pipeline_config.build(labels=labels)

        # build explainer
        multi_target = (
            self.config.explanation_target_strategy == ExplanationTargetStrategy.all
        )

        # build explainer
        self._explainer = self.config.explainer_config.build(
            model=self._model_pipeline._model, multi_target=multi_target
        )

        # build feature segmentor
        self._feature_segmentor = self.config.feature_segmentor_config.build()

        # build baselines generator
        self._baseline_generator = self.config.baseline_generator_config.build()

        # get possible explainer args
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        self._explainer_args = inspect.signature(
            self._explainer.explain
        ).parameters.keys()

    @property
    def ops(self) -> Any:
        return self._model_pipeline.ops

    @abstractmethod
    def _model_forward(self, batch: T_TensorDataModel) -> Any:
        """Forward pass through the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _target(
        self, batch: T_TensorDataModel, model_outputs: Any
    ) -> ExplanationTargetType | list[ExplanationTargetType]:
        """Prepare the explanation target based on the strategy."""
        pass

    @abstractmethod
    def _explained_inputs(
        self, batch: T_TensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        """Prepare the input features for the explainer."""
        pass

    def _additional_forward_args(self, batch: T_TensorDataModel) -> Any | None:
        """Prepare any additional forward arguments for the explainer."""
        return None

    def _baselines(
        self,
        explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor],
        train_baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        """Generate baselines for the explainer."""
        if isinstance(explained_inputs, torch.Tensor):
            if train_baselines is not None and not isinstance(
                train_baselines, torch.Tensor
            ):
                raise ValueError(
                    "Explained inputs are a tensor, but train_baselines is not a tensor. Found: "
                    f"{explained_inputs=},"
                    f"{train_baselines=}"
                )
            baselines = (
                train_baselines
                if train_baselines is not None
                else self._baseline_generator(explained_inputs)
            )
        else:
            if train_baselines is not None and not isinstance(
                train_baselines, OrderedDict
            ):
                raise ValueError(
                    "Explained inputs are an OrderedDict, but train_baselines is not an OrderedDict. Found: "
                    f"{explained_inputs=},"
                    f"{train_baselines=}"
                )
            baselines = OrderedDict()
            for key, tensor in explained_inputs.items():
                baselines[key] = (
                    train_baselines[key]
                    if train_baselines is not None
                    else self._baseline_generator(tensor)
                )
        return baselines

    def _feature_mask(
        self, explained_inputs: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> Any:
        """Generate feature mask using the feature segmentor."""
        if isinstance(explained_inputs, torch.Tensor):
            return self._feature_segmentor(explained_inputs)
        else:
            return OrderedDict(
                {
                    key: self._feature_segmentor(tensor)
                    for key, tensor in explained_inputs.items()
                }
            )

    def _prepare_explanation_inputs(
        self,
        batch: T_TensorDataModel,
        target: ExplanationTargetType | list[ExplanationTargetType],
        train_baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> ExplanationInputs:
        """Prepare the inputs for the explainer step."""
        # prepare explained inputs
        inputs = self._explained_inputs(batch)

        # prepare additional forward args
        additional_forward_args = self._additional_forward_args(batch) or ()

        # prepare baselines
        baselines = self._baselines(
            explained_inputs=inputs, train_baselines=train_baselines
        )

        # prepare feature mask
        feature_mask = self._feature_mask(explained_inputs=inputs)

        # prepare explanation inputs
        return ExplanationInputs(
            sample_id=batch.metadata.sample_id,
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            baselines=baselines,
            feature_mask=feature_mask,
            target=target,
            frozen_features=None,
        )

    def _explainer_forward(
        self, explanation_inputs: ExplanationInputs
    ) -> ExplanationState:
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        kwargs = {}
        for arg in self._explainer_args:
            kwargs[arg] = getattr(explanation_inputs, arg)

        logger.debug(
            f"Explainer forward with args: {', '.join(f'{k}={v}' for k, v in kwargs.items())}"
        )

        if self.config.iterative_computation and isinstance(
            explanation_inputs.target, list
        ):
            # disable multi-target for iterative computation
            self._explainer.multi_target = False

            target = explanation_inputs.target
            per_target_explanations = []
            for t in tqdm(target, desc="Computing explanations per target"):
                kwargs = {**kwargs, "target": t}
                curr_explanation = self._explainer.explain(**kwargs)
                per_target_explanations.append(curr_explanation.explanations)

            # re-enable multi-target
            self._explainer.multi_target = True
            return ExplanationState(
                explanation_inputs=explanation_inputs,
                explanations=per_target_explanations,
            )
        else:
            return self._explainer.explain(**kwargs)

    def explanation_step(
        self,
        batch: T_TensorDataModel,
        train_baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> ExplanationState:
        model_outputs = self._model_forward(batch)
        target = self._target(batch=batch, model_outputs=model_outputs)
        explainer_step_inputs = self._prepare_explanation_inputs(
            batch=batch, target=target, train_baselines=train_baselines
        )
        return self._explainer_forward(explanation_inputs=explainer_step_inputs)

    def build_metrics(self, device: torch.device | str = "cpu") -> dict[str, Metric]:
        return {}

    # def build_metrics(
    #     self,
    #     stage: Literal[train, validation, test],
    #     device: torch.device | str = "cpu",
    # ) -> dict[str, Metric]:
    #     if self.config.metrics is None:
    #         return {}
    #     assert self._labels.classification is not None, (
    #         "Labels must be provided for classification tasks."
    #     )
    #     metrics = {}
    #     for metric_config in self.config.metrics:
    #         logger.info(f"Building metric: {metric_config}")
    #         metric = metric_config.build(
    #             device=device, num_classes=len(self._labels.classification), stage=stage
    #         )
    #         metrics[metric_config.name] = metric
    #     return metrics
