from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

from atria_insights.engines._feature_generation_step import FeatureGenerationStep
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline

if TYPE_CHECKING:
    from ignite.engine import Engine, State

logger = get_logger(__name__)


class FeatureGenerationEngineConfig(EngineConfig):
    max_features: int = 100


class FeatureGenerationEngineDependencies(EngineDependencies):
    x_model_pipeline: ExplainableModelPipeline
    feature_file_name: str


class FeatureGenerationEngine(
    EngineBase[FeatureGenerationEngineConfig, FeatureGenerationEngineDependencies]
):
    def _build_engine_step(self) -> EngineStep:
        return FeatureGenerationStep(
            x_model_pipeline=self._deps.x_model_pipeline,
            device=self._deps.device,
            with_amp=False,
            cache_dir=self._deps.output_dir,
            file_name=self._deps.feature_file_name,
        )

    def _attach_handlers(self) -> None:
        from ignite.engine import Events

        # configure engine
        self._setup_test_run()
        self._attach_progress_bar()

        @self._engine.on(Events.TERMINATE | Events.INTERRUPT)
        def on_terminate(engine: Engine) -> None:
            logger.info(
                f"Engine [{self.__class__.__name__}] terminated after {engine.state.epoch} epochs."
            )

        # add handler for exception
        @self._engine.on(Events.EXCEPTION_RAISED)
        def on_exception(exception: Exception) -> None:
            raise exception

    def run(
        self, checkpoint_path: str | Path | None = None, overwrite: bool = False
    ) -> State | None:
        from atria_ml.training.engines.utilities import FixedBatchIterator

        if not overwrite:
            # check if features already exist
            feature_file_path = (
                Path(self._deps.output_dir) / self._deps.feature_file_name
            )
            if feature_file_path.exists():
                logger.info(
                    f"Features file already exists at {feature_file_path}. Skipping feature generation."
                )
                return None

        # run engine
        if self._deps.output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._deps.dataloader.batch_size}] and output_dir: {self._deps.output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        # move model pipeline to device
        self._deps.model_pipeline.ops.to_device(self._deps.device)

        return self._engine.run(
            (
                FixedBatchIterator(
                    self._deps.dataloader, self._deps.dataloader.batch_size
                )
                if self._config.use_fixed_batch_iterator
                else self._deps.dataloader
            ),
            max_epochs=self._config.max_epochs,
            epoch_length=round(
                self._config.max_features / self._deps.dataloader.batch_size
            ),
        )
