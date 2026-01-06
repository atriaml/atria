from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies
from ignite.engine import State

from atria_insights.perturbation_robustness._evaluator_step import (
    PerturbationRobustnessEvaluatorStep,
)

logger = get_logger(__name__)


class PerturbationRobustnessEvaluatorEngineConfig(EngineConfig):
    pass


class PerturbationRobustnessEvaluatorEngineDependencies(EngineDependencies):
    perturbation_transform: Callable


class PerturbationRobustnessEvaluatorEngine(
    EngineBase[
        PerturbationRobustnessEvaluatorEngineConfig,
        PerturbationRobustnessEvaluatorEngineDependencies,
    ]
):
    def __init__(self, config, deps):
        super().__init__(config, deps)
        self._last_checkpoint_path = None

    def _build_engine_step(self) -> EngineStep:
        return PerturbationRobustnessEvaluatorStep(
            model_pipeline=self._deps.model_pipeline,
            perturbation_transform=self._deps.perturbation_transform,
            device=self._deps.device,
        )

    def _rebuild_engine(self, perturbation_transform: Callable) -> None:
        # rebuild step
        self._engine_step = PerturbationRobustnessEvaluatorStep(
            model_pipeline=self._deps.model_pipeline,
            perturbation_transform=perturbation_transform,
            device=self._deps.device,
        )

        # initialize the Ignite engine
        self._engine = self._initialize_ignite_engine(engine_step=self._engine_step)
        self._attach_handlers()

    def run(
        self,
        perturbation_transform: Callable,
        checkpoint_path: str | Path | None = None,
    ) -> State:
        from atria_ml.training.engines.utilities import FixedBatchIterator

        # run engine
        if self._deps.output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._deps.dataloader.batch_size}] and output_dir: {self._deps.output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        # move model pipeline to device
        self._deps.model_pipeline.ops.to_device(self._deps.device)

        # rebuild engine with new perturbation transform
        self._rebuild_engine(perturbation_transform=perturbation_transform)

        # load checkpoint if provided
        if (
            checkpoint_path is not None
            and checkpoint_path != self._last_checkpoint_path
        ):
            logger.debug(
                f"Loading checkpoint from path: {checkpoint_path} for evaluator engine."
            )
            self._load_checkpoint(checkpoint_path=checkpoint_path)
            self._last_checkpoint_path = checkpoint_path

        return self._engine.run(
            (
                FixedBatchIterator(
                    self._deps.dataloader, self._deps.dataloader.batch_size
                )
                if self._config.use_fixed_batch_iterator
                else self._deps.dataloader
            ),
            max_epochs=self._config.max_epochs,
            epoch_length=self._config.epoch_length,
        )
