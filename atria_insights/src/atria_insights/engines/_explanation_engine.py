from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies
from ignite.engine import Engine, Events

from atria_insights.engines._events import MetricUpdateEvents
from atria_insights.engines._explanation_step import ExplanationStep
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline

if TYPE_CHECKING:
    from ignite.engine import Engine

logger = get_logger(__name__)


class ExplanationEngineConfig(EngineConfig):
    enable_outputs_caching: bool = True
    compute_metrics: bool = True


class ExplanationEngineDependencies(EngineDependencies):
    x_model_pipeline: ExplainableModelPipeline


class ExplanationEngine(
    EngineBase[ExplanationEngineConfig, ExplanationEngineDependencies]
):
    def __init__(
        self, config: ExplanationEngineConfig, deps: ExplanationEngineDependencies
    ) -> None:
        self._config = config
        self._deps = deps
        self._engine_step, self._engine = self._build_engine()
        if self._config.compute_metrics:
            self._x_metrics = self._deps.x_model_pipeline.build_metrics(
                device=self._deps.device
            )
        else:
            self._x_metrics = None
        self._register_events()
        self._attach_handlers()

    def _build_engine_step(self) -> EngineStep:
        return ExplanationStep(
            x_model_pipeline=self._deps.x_model_pipeline, device=self._deps.device
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

    def _register_events(self) -> None:
        self._engine.register_events(
            *MetricUpdateEvents,  # type: ignore[arg-type]
            event_to_attr={
                MetricUpdateEvents.X_METRIC_STARTED: "x_metric_started",
                MetricUpdateEvents.X_METRIC_COMPLETED: "x_metric_completed",
            },
        )

    def _attach_progress_bar(self) -> None:
        import ignite.distributed as idist
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
            )

            @self._engine.on(MetricUpdateEvents.X_METRIC_STARTED)
            def on_metric_update(engine: Engine) -> None:
                if progress_bar.pbar is not None:
                    progress_bar.pbar.set_postfix(
                        {"computing x_metric": engine.state.x_metric_started}
                    )

            @self._engine.on(Events.TERMINATE | Events.INTERRUPT)
            def on_terminate(engine: Engine) -> None:
                progress_bar.close()
