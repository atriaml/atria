from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from atria_registry import ModuleConfig

from atria_insights.baseline_generators._base import BaselineGenerator
from atria_insights.data_types._common import BaselineStrategy


class SimpleBaselineGeneratorConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.baseline_generators._simple.SimpleBaselineGenerator"
    )
    type: Literal["simple"] = "simple"
    baseline_strategy: BaselineStrategy = BaselineStrategy.zeros
    baselines_fixed_value: float = 0.0


class SimpleBaselineGenerator(BaselineGenerator[SimpleBaselineGeneratorConfig]):
    __config__ = SimpleBaselineGeneratorConfig

    def _prepare_baselines_from_type(
        self,
        inp: torch.Tensor,
        baselines_type: Literal["zeros", "ones", "batch_mean", "random", "fixed"],
        fixed_value: float = 0.5,
    ) -> torch.Tensor:
        if baselines_type == "zeros":
            return torch.zeros_like(inp)
        elif baselines_type == "ones":
            return torch.ones_like(inp)
        elif baselines_type == "batch_mean":
            return torch.full_like(inp, inp.mean().item())
        elif baselines_type == "fixed":
            return torch.full_like(inp, fixed_value)
        elif baselines_type == "random":
            return torch.rand_like(inp)
        else:
            raise ValueError(
                f"Unsupported baselines_type: {baselines_type}. Supported types are 'zeros' and 'random'."
            )

    def __call__(  # type: ignore[override]
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        if isinstance(inputs, OrderedDict):
            baselines = OrderedDict()
            for key, tensor in inputs.items():
                baselines[key] = self._prepare_baselines_from_type(
                    tensor,
                    baselines_type=self.config.baseline_strategy,  # type: ignore[arg-type]
                    fixed_value=self.config.baselines_fixed_value,
                )
            return baselines
        else:
            return self._prepare_baselines_from_type(
                inputs,
                baselines_type=self.config.baseline_strategy,  # type: ignore[arg-type]
                fixed_value=self.config.baselines_fixed_value,
            )
