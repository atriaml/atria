from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic

from atria_models.core.model_pipelines._common import (
    _DEFAULT_OPTIMIZER_PARAMETERS_KEY,
    T_ModelPipelineConfig,
)

if TYPE_CHECKING:
    import torch

    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline


class ModelPipelineOps(Generic[T_ModelPipelineConfig]):
    def __init__(self, model_pipeline: ModelPipeline[T_ModelPipelineConfig]) -> None:
        self._model_pipeline = model_pipeline

    def to_device(
        self, device: str | torch.device, sync_bn: bool = False
    ) -> ModelPipeline[T_ModelPipelineConfig]:
        import torch
        from torch import nn

        from atria_models.utilities._ddp_model_proxy import ModuleProxyWrapper
        from atria_models.utilities._nn_modules import _auto_model

        self._model_pipeline._model = _auto_model(
            device=torch.device(device),
            model=self._model_pipeline._model,
            sync_bn=sync_bn,
        )

        if isinstance(
            self._model_pipeline._model,
            nn.parallel.DistributedDataParallel | nn.DataParallel,
        ):
            self._model_pipeline._model = ModuleProxyWrapper(
                self._model_pipeline._model
            )

        return self._model_pipeline

    def train(self) -> ModelPipeline[T_ModelPipelineConfig]:
        self._model_pipeline._model.train()
        return self._model_pipeline

    def eval(self) -> ModelPipeline[T_ModelPipelineConfig]:
        self._model_pipeline._model.eval()
        return self._model_pipeline

    def half(self) -> ModelPipeline[T_ModelPipelineConfig]:
        self._model_pipeline._model.half()
        return self._model_pipeline

    def get_trainable_parameters(self) -> dict[str, Iterator[torch.nn.Parameter]]:
        return {
            _DEFAULT_OPTIMIZER_PARAMETERS_KEY: self._model_pipeline._model.parameters()
        }

    def summarize(self):
        from torch import nn
        from torchinfo import summary

        nn_module_dict = nn.ModuleDict()
        for k, v in self._model_pipeline.__dict__.items():
            if isinstance(v, nn.Module):
                nn_module_dict.add_module(k, v)
        return str(summary(nn_module_dict, verbose=0, depth=3))
