from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

from atria_insights.engines._explanation_step import ExplanationStep
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine

logger = get_logger(__name__)


class ExplanationEngineConfig(EngineConfig):
    enable_outputs_caching: bool = True


class ExplanationEngineDependencies(EngineDependencies):
    x_model_pipeline: ExplainableModelPipeline
    train_baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None


class ExplanationEngine(
    EngineBase[ExplanationEngineConfig, ExplanationEngineDependencies]
):
    def __init__(
        self, config: ExplanationEngineConfig, deps: ExplanationEngineDependencies
    ) -> None:
        self._config = config
        self._deps = deps
        self._engine_step, self._engine = self._build_engine()
        self._x_metrics = self._deps.x_model_pipeline.build_metrics(
            # stage=self._engine_step.name, device=self._deps.device
        )
        self._attach_handlers()

    def _build_engine_step(self) -> EngineStep:
        return ExplanationStep(
            x_model_pipeline=self._deps.x_model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            train_baselines=self._deps.train_baselines,
            enable_outputs_caching=self._config.enable_outputs_caching,
            cache_dir=self._deps.output_dir,
        )

    def _attach_metrics(self) -> None:
        import ignite.distributed as idist
        import torch
        from ignite.metrics import GpuInfo
        from ignite.metrics.metric import RunningBatchWise

        if self._config.logging.log_gpu_stats:
            if idist.device() != torch.device("cpu"):
                GpuInfo().attach(self._engine, name="gpu")

        if self._x_metrics is not None and len(self._x_metrics) > 0:
            for metric_name, metric in self._x_metrics.items():
                logger.info(
                    f"Attaching metric '{metric_name}' to engine '{self.__class__.__name__}'"
                )
                metric.attach(
                    self._engine,
                    (
                        f"{self._engine_step.name}/{metric_name}"
                        if self._config.metric_logging_prefix is None
                        else f"{self._engine_step.name}/{self._config.metric_logging_prefix}/{metric_name}"
                    ),
                    usage=RunningBatchWise(),
                )

    def _attach_progress_bar(self) -> None:
        import ignite.distributed as idist
        from ignite.engine import Events
        from ignite.handlers import ProgressBar

        # initialize the progress bar
        progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.name}]", persist=True
        )

        if idist.get_rank() == 0:
            progress_bar.attach(
                self._engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=self._config.logging.refresh_rate
                ),
                metric_names="all",
            )

            @self._engine.on(Events.TERMINATE | Events.INTERRUPT)
            def on_terminate(engine: Engine) -> None:
                progress_bar.close()
