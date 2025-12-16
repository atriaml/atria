from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    ExplanationPipeline,
)

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine

logger = get_logger(__name__)


class ExplanationEngineConfig(EngineConfig):
    save_outputs_to_fs: bool = False


class ExplanationEngineDependencies(EngineDependencies):
    explanation_pipeline: ExplanationPipeline
    train_baselines: dict[str, torch.Tensor]


class NoCheckpointFoundError(Exception):
    pass


class ExplanationEngine(
    EngineBase[ExplanationEngineConfig, ExplanationEngineDependencies]
):
    def __init__(
        self, config: ExplanationEngineConfig, deps: ExplanationEngineDependencies
    ) -> None:
        super().__init__(config=config, deps=deps)

    def _build_engine_step(self) -> EngineStep:
        return ExplanationStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            train_baselines=self._deps.train_baselines,
        )

    def _attach_metrics(self) -> None:
        import ignite.distributed as idist
        import torch
        from ignite.metrics import GpuInfo
        from ignite.metrics.metric import RunningBatchWise

        if self._config.logging.log_gpu_stats:
            if idist.device() != torch.device("cpu"):
                GpuInfo().attach(self._engine, name="gpu")

        if self._metrics is not None and len(self._metrics) > 0:
            for metric_name, metric in self._metrics.items():
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
