"""
Evaluation Engines Module

This module defines various engines for evaluation, testing, visualization, and feature extraction
in the Atria framework. These engines manage specific tasks such as validation, testing, and
visualization during the training process.

Classes:
    - ValidationEngine: Engine for managing validation tasks.
    - TestEngine: Engine for managing testing tasks.
    - VisualizationEngine: Engine for managing visualization tasks.
    - PredictionEngine: Engine for managing prediction tasks.
    - FeatureExtractorEngine: Engine for extracting features from datasets.

Dependencies:
    - ignite: For distributed training and engine management.
    - atria_core.logger: For logging utilities.
    - atria_ml.registry: For registering engines.
    - atria_ml.training.configs: For various training configurations.
    - atria_ml.training.engines: For engine utilities and steps.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger

from atria_ml.registry import ENGINE
from atria_ml.training.configs.logging_config import LoggingConfig
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.engine_steps.base import BaseEngineStep
from atria_ml.training.engines.engine_steps.evaluation import (
    FeatureExtractorStep,
    InferenceStep,
    TestStep,
    ValidationStep,
    VisualizationStep,
)

if TYPE_CHECKING:
    import torch
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from ignite.engine import Engine, State

    from atria_ml.training.handlers.ema_handler import AtriaEMAHandler


logger = get_logger(__name__)


@ENGINE.register("default_validation_engine")
class ValidationEngine(AtriaEngine):
    """
    Validation Engine

    This engine is responsible for managing validation tasks during the training process.
    It supports various configurations for different types of validation tasks.

    Attributes:
        _REGISTRY_CONFIGS (ClassVar[Type[dict]]): Registry configurations for validation tasks.
        _validate_every_n_epochs (Optional[float]): Frequency of validation in terms of epochs.
        _validate_on_start (bool): Whether to validate at the start of training.
        _min_train_epochs_for_best (Optional[int]): Minimum training epochs for best validation.
        _use_ema_for_val (bool): Whether to use EMA for validation.
        _ema_handler (Optional[AtriaEMAHandler]): EMA handler for validation.
        _parent_engine (Optional[Engine]): Parent engine to attach validation engine.
    """

    def __init__(
        self,
        epoch_length: int | None = None,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig | None = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
        validate_every_n_epochs: float | None = 1,
        validate_on_start: bool = True,
        min_train_epochs_for_best: int | None = 1,
        use_ema_for_val: bool = False,
        with_amp: bool = False,
    ):
        """
        Initialize the ValidationEngine.

        Args:
            engine_step (ValidationStep): Validation step to attach to the engine.
            epoch_length (Optional[int]): Length of each epoch.
            outputs_to_running_avg (Optional[List[str]]): Outputs to calculate running average.
            logging (Optional[LoggingConfig]): Logging configuration.
            metric_logging_prefix (Optional[str]): Prefix for metric logging.
            test_run (bool): Whether to run in test mode.
            use_fixed_batch_iterator (bool): Whether to use fixed batch iterator.
            checkpoints_dir (str): Directory for saving checkpoints.
            validate_every_n_epochs (Optional[float]): Frequency of validation in epochs.
            validate_on_start (bool): Whether to validate at the start of training.
            min_train_epochs_for_best (Optional[int]): Minimum training epochs for best validation.
            use_ema_for_val (bool): Whether to use EMA for validation.
            with_amp (bool): Whether to use automatic mixed precision.
        """
        super().__init__(
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )
        self._validate_every_n_epochs = validate_every_n_epochs
        self._validate_on_start = validate_on_start
        self._min_train_epochs_for_best = min_train_epochs_for_best
        self._use_ema_for_val = use_ema_for_val
        self._with_amp = with_amp
        self._ema_handler = None
        self._parent_engine = None

    def _setup_engine_step(self):
        return ValidationStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def _configure_engine(self, engine: Engine) -> Engine:
        """
        Configure the validation engine.

        Args:
            engine (Engine): Ignite engine instance.
        """
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_running_avg_logging(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
        self._attach_event_handlers(engine=engine)
        return engine

    def _configure_tb_logger(self, engine: Engine):
        """
        Configure TensorBoard logger for validation engine.

        Args:
            engine (Engine): Ignite engine instance.
        """
        import ignite.distributed as idist
        from ignite.contrib.handlers import global_step_from_engine
        from ignite.engine import Events

        if (
            idist.get_rank() == 0
            and self._tb_logger is not None
            and self._logging.log_to_tb
        ):
            assert self._parent_engine is not None, (
                "Training engine is not set. You must call `attach_to_engine` first."
            )
            # attach tb logger to validation engine
            self._tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
                global_step_transform=global_step_from_engine(self._parent_engine),
            )

    def _initialize_ignite_engine(self) -> Engine:
        """
        Initialize the validation engine.

        Returns:
            Engine: Ignite engine instance.
        """
        self._engine_step.attach_parent_engine(self._parent_engine)
        return super()._initialize_ignite_engine()

    def attach_to_engine(
        self,
        parent_engine: Engine,
        steps_per_epoch: int,
        ema_handler: AtriaEMAHandler | None = None,
    ):
        """
        Attach the validation engine to a parent engine.

        Args:
            parent_engine (Engine): Parent engine instance.
            steps_per_epoch (int): Number of steps per epoch.
            ema_handler (Optional[AtriaEMAHandler]): EMA handler for validation.
        """
        from ignite.engine import Events

        if self._validate_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=int(self._validate_every_n_epochs))
            cond = cond | Events.COMPLETED
            if self._validate_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(cond, self.run)
        else:
            cond = Events.ITERATION_COMPLETED(
                every=int(self._validate_every_n_epochs * steps_per_epoch)
            )
            cond = cond | Events.COMPLETED
            if self._validate_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(cond, self.run)

        self._parent_engine = parent_engine
        self._ema_handler = ema_handler

    def run(self) -> Engine:
        """
        Run the validation engine.

        Returns:
            Engine: Ignite engine instance.
        """
        if self._use_ema_for_val:
            if self._ema_handler is None:
                logger.warning(
                    "EMA handler is not set. You must pass an "
                    "EMA handler to `attach_to_engine` to use ema for validation."
                )
            else:
                self._ema_handler.swap_params()
        super().run()
        if self._use_ema_for_val:
            if self._ema_handler is not None:
                self._ema_handler.swap_params()


@ENGINE.register("default_test_engine")
class TestEngine(AtriaEngine):
    """
    Test Engine

    This engine is responsible for managing testing tasks during the training process.
    It supports various configurations for different types of testing tasks.

    Attributes:
        _REGISTRY_CONFIGS (ClassVar[Type[dict]]): Registry configurations for testing tasks.
        _save_model_outputs_to_disk (bool): Whether to save model outputs to disk.
        _checkpoint_types (Optional[List[str]]): Types of checkpoints to load.
    """

    def __init__(
        self,
        epoch_length: int | None = None,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
        save_model_outputs_to_disk: bool = False,
        checkpoint_types: list[str] | None = None,
        with_amp: bool = False,
    ):
        """
        Initialize the TestEngine.

        Args:
            engine_step (TestStep): Test step to attach to the engine.
            epoch_length (Optional[int]): Length of each epoch.
            outputs_to_running_avg (Optional[List[str]]): Outputs to calculate running average.
            logging (LoggingConfig): Logging configuration.
            metric_logging_prefix (Optional[str]): Prefix for metric logging.
            test_run (bool): Whether to run in test mode.
            use_fixed_batch_iterator (bool): Whether to use fixed batch iterator.
            checkpoints_dir (str): Directory for saving checkpoints.
            save_model_outputs_to_disk (bool): Whether to save model outputs to disk.
            checkpoint_types (Optional[List[str]]): Types of checkpoints to load.
            with_amp (bool): Whether to use automatic mixed precision.
        """
        super().__init__(
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )
        self._save_model_outputs_to_disk = save_model_outputs_to_disk
        self._checkpoint_types = checkpoint_types or ["last", "best"]
        self._with_amp = with_amp
        for key in self._checkpoint_types:
            assert key in ["last", "best"], (
                f"Checkpoint type {key} is not supported. Possible types are ['last', 'best']"
            )

    def _setup_engine_step(self):
        return TestStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def _configure_model_forward_saver(self, engine: Engine, output_name: str):
        """
        Configure model forward saver for saving outputs to disk.

        Args:
            engine (Engine): Ignite engine instance.
            output_name (str): Path to the output file.
        """
        from ignite.engine import Events

        from atria_ml.training.utilities.model_output_saver import ModelOutputSaver

        if self._save_model_outputs_to_disk:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                ModelOutputSaver(output_dir=self._output_dir, output_name=output_name),
            )

    def _find_checkpoint_files_in_output_dir(self) -> list[str]:
        from atria_ml.training.utilities.checkpoints import find_test_checkpoint

        checkpoint_dir = Path(self._output_dir) / self._checkpoints_dir
        checkpoint_files = []
        for load_best in [False, True]:
            checkpoint = find_test_checkpoint(checkpoint_dir, load_best=load_best)
            if checkpoint is not None:
                checkpoint_files.append(checkpoint)
        return checkpoint_files

    def _load_checkpoint(self, checkpoint_file: str):
        """
        Load checkpoint for testing.

        Args:
            checkpoint_type (str): Type of checkpoint to load.

        Returns:
            str: Path to the loaded checkpoint file.
        """
        import torch
        from ignite.handlers import Checkpoint

        test_checkpoint = torch.load(checkpoint_file, map_location="cpu")
        Checkpoint.load_objects(
            to_load={"model_pipeline": self._model_pipeline}, checkpoint=test_checkpoint
        )

        return checkpoint_file

    def run(self, test_checkpoint: str | None = None) -> dict[str, State]:
        """
        Run the test engine.

        Returns:
            dict[str, State]: Output states for each checkpoint type.
        """

        # find checkpoint files
        if test_checkpoint is None:
            checkpoint_file_paths = self._find_checkpoint_files_in_output_dir()
        else:
            checkpoint_file_paths = [test_checkpoint]

        if len(checkpoint_file_paths) == 0:
            logger.warning(
                "No checkpoint files found for testing. "
                "Running test with pretrained model without loading any checkpoint."
            )
            return super().run()

        output_states = {}
        for checkpoint_file_path in checkpoint_file_paths:
            logger.info(
                f"Checkpoint detected, running test with checkpoint: {checkpoint_file_path}"
            )
            # configure checkpoint
            if checkpoint_file_path is not None:
                self._load_checkpoint(checkpoint_file_path)

            checkpoint_name = Path(checkpoint_file_path).name.replace("=", "_")
            if self._metric_logging_prefix:
                self._metric_logging_prefix += "/" + checkpoint_name
            else:
                self._metric_logging_prefix = checkpoint_name

            # reinitialize engine
            self._engine = self._initialize_ignite_engine()

            # run test engine
            output_states[checkpoint_name] = super().run()
        return output_states


@ENGINE.register("default_visualization_engine")
class VisualizationEngine(AtriaEngine):
    """
    Visualization Engine

    This engine is responsible for managing visualization tasks during the training process.

    Attributes:
        _visualize_every_n_epochs (Optional[float]): Frequency of visualization in terms of epochs.
        _visualize_on_start (bool): Whether to visualize at the start of training.
        _use_ema_for_visualize (bool): Whether to use EMA for visualization.
        _ema_handler (Optional[AtriaEMAHandler]): EMA handler for visualization.
        _parent_engine (Optional[Engine]): Parent engine to attach visualization engine.
    """

    def __init__(
        self,
        epoch_length: int | None = 1,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
        visualize_every_n_epochs: float | None = 1,
        visualize_on_start: bool = False,
        use_ema_for_visualize: bool = False,
        with_amp: bool = False,
    ):
        """
        Initialize the VisualizationEngine.

        Args:
            epoch_length (Optional[int]): Length of each epoch.
            outputs_to_running_avg (Optional[List[str]]): Outputs to calculate running average.
            logging (LoggingConfig): Logging configuration.
            metric_logging_prefix (Optional[str]): Prefix for metric logging.
            test_run (bool): Whether to run in test mode.
            use_fixed_batch_iterator (bool): Whether to use fixed batch iterator.
            checkpoints_dir (str): Directory for saving checkpoints.
            visualize_every_n_epochs (Optional[float]): Frequency of visualization in epochs.
            visualize_on_start (bool): Whether to visualize at the start of training.
            use_ema_for_visualize (bool): Whether to use EMA for visualization.
            with_amp (bool): Whether to use automatic mixed precision.
        """
        super().__init__(
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )
        self._visualize_every_n_epochs = visualize_every_n_epochs
        self._visualize_on_start = visualize_on_start
        self._use_ema_for_visualize = use_ema_for_visualize
        self._with_amp = with_amp
        self._ema_handler = None
        self._parent_engine = None

    def _setup_engine_step(self):
        return VisualizationStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def _initialize_ignite_engine(self) -> Engine:
        """
        Initialize the visualization engine.

        Returns:
            Engine: Ignite engine instance.
        """
        self._engine_step.attach_parent_engine(self._parent_engine)
        return super()._initialize_ignite_engine()

    def _configure_engine(self, engine: Engine) -> Engine:
        """
        Configure the visualization engine.

        Args:
            engine (Engine): Ignite engine instance.
        """
        self._configure_test_run(engine=engine)
        self._configure_progress_bar(engine=engine)
        return engine

    def _configure_progress_bar(self, engine: Engine) -> None:
        """
        Configure progress bar for visualization engine.

        Args:
            engine (Engine): Ignite engine instance.
        """
        import ignite.distributed as idist
        from ignite.engine import Events

        if idist.get_rank() == 0:
            self._progress_bar.attach(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=self._logging.refresh_rate),
            )

    def attach_to_engine(
        self,
        parent_engine: Engine,
        steps_per_epoch: int,
        ema_handler: AtriaEMAHandler | None = None,
    ):
        """
        Attach the visualization engine to a parent engine.

        Args:
            parent_engine (Engine): Parent engine instance.
            steps_per_epoch (int): Number of steps per epoch.
            ema_handler (Optional[AtriaEMAHandler]): EMA handler for visualization.
        """
        from ignite.engine import Events

        if self._visualize_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=int(self._visualize_every_n_epochs))
            cond = cond | Events.COMPLETED
            if self._visualize_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(cond, self.run)
        else:
            cond = Events.ITERATION_COMPLETED(
                every=int(self._visualize_every_n_epochs * steps_per_epoch)
            )
            cond = cond | Events.COMPLETED
            if self._visualize_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(cond, self.run)

        self._parent_engine = parent_engine
        self._ema_handler = ema_handler

    def run(self) -> Engine:
        """
        Run the visualization engine.

        Returns:
            Engine: Ignite engine instance.
        """
        if self._use_ema_for_visualize:
            assert self._ema_handler is not None, (
                "EMA handler is not set. You must pass an "
                "EMA handler to `attach_to_engine` to use ema for visualization."
            )
            self._ema_handler.swap_params()
        super().run()
        if self._use_ema_for_visualize:
            assert self._ema_handler is not None, (
                "EMA handler is not set. You must pass an "
                "EMA handler to `attach_to_engine` to use ema for visualization."
            )
            self._ema_handler.swap_params()


@ENGINE.register("default_inference_engine")
class InferenceEngine(AtriaEngine):
    """
    Test InferenceEngine

    This engine is responsible for managing inference tasks during the training process.
    It supports various configurations for different types of inference tasks.

    Attributes:
        _REGISTRY_CONFIGS (ClassVar[Type[dict]]): Registry configurations for testing tasks.
    """

    def __init__(
        self,
        epoch_length: int | None = None,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        with_amp: bool = False,
    ):
        """
        Initialize the InferenceEngine.

        Args:
            engine_step (InferenceStep): Inference step to attach to the engine.
            epoch_length (Optional[int]): Length of each epoch.
            outputs_to_running_avg (Optional[List[str]]): Outputs to calculate running average.
            logging (LoggingConfig): Logging configuration.
            metric_logging_prefix (Optional[str]): Prefix for metric logging.
            test_run (bool): Whether to run in test mode.
            use_fixed_batch_iterator (bool): Whether to use fixed batch iterator.
        """
        super().__init__(
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
        )
        self._with_amp = with_amp

    def _setup_engine_step(self) -> BaseEngineStep:
        """
        Setup the engine step for inference.

        Returns:
            BaseEngineStep: The engine step instance.
        """
        return InferenceStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def build(
        self, model_pipeline: AtriaModelPipeline, device: str | torch.device
    ) -> AtriaEngine:
        """
        Build the engine with the specified configurations.

        Args:
            model_pipeline (AtriaModelPipeline): Model pipeline to use.
            device (Union[str, torch.device]): Device to run the engine on.

        Returns:
            AtriaEngine: The configured engine instance.
        """
        import torch
        from ignite.handlers import ProgressBar

        self._model_pipeline = model_pipeline
        self._device = torch.device(device)

        # initialize the engine step
        self._engine_step = self._setup_engine_step()

        # initialize the progress bar
        self._progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.stage}]", persist=True
        )

        # move task module models to device
        self._model_pipeline.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize engine
        self._engine = self._initialize_ignite_engine()

        return self

    def _configure_engine(self, engine: Engine) -> Engine:
        """
        Configure the engine with handlers and settings.

        Args:
            engine (Engine): The engine instance to configure.
        """
        self._configure_metrics(engine=engine)
        self._configure_progress_bar(engine=engine)
        return engine

    def run(self, dataloader: Iterator) -> dict[str, State]:
        """
        Run the test engine.

        Returns:
            dict[str, State]: Output states for each checkpoint type.
        """
        from atria_ml.training.engines.utilities import FixedBatchIterator

        # run inference
        return self._engine.run(
            (
                FixedBatchIterator(dataloader, dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else dataloader
            ),
            max_epochs=1,
        )


@ENGINE.register("feature_extractor_engine/default")
class FeatureExtractorEngine(AtriaEngine):
    """
    Feature Extractor Engine

    This engine is responsible for extracting features from datasets during the training process.

    Attributes:
        _features_key (str): Key for features to extract.
        _cache_file_name (str): Name of the cache file for features.
        _max_shard_size (int): Maximum size of each shard for feature extraction.
    """

    def __init__(
        self,
        engine_step: FeatureExtractorStep,
        features_key: str,
        cache_file_name: str,
        test_run: bool = False,
        max_shard_size: int = 100000,
        with_amp: bool = False,
    ):
        """
        Initialize the FeatureExtractorEngine.

        Args:
            engine_step (FeatureExtractorStep): Feature extraction step to attach to the engine.
            features_key (str): Key for features to extract.
            cache_file_name (str): Name of the cache file for features.
            test_run (bool): Whether to run in test mode.
            max_shard_size (int): Maximum size of each shard for feature extraction.
            with_amp (bool): Whether to use automatic mixed precision.
        """
        super().__init__(
            engine_step=engine_step,
            tb_logger=None,
            max_epochs=1,
            epoch_length=None,
            outputs_to_running_avg=[],
            logging=LoggingConfig(logging_steps=1, refresh_rate=1),
            metric_logging_prefix=None,
            test_run=test_run,
        )
        self._features_key = features_key
        self._cache_file_name = cache_file_name
        self._max_shard_size = max_shard_size
        self._with_amp = with_amp

    def _setup_engine_step(self) -> FeatureExtractorStep:
        """
        Setup the engine step for feature extraction.

        Returns:
            FeatureExtractorStep: The engine step instance.
        """
        return FeatureExtractorStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            test_run=self._test_run,
            with_amp=self._with_amp,
        )

    def _prepare_output_file_pattern(self, split: str, proc: int = 0) -> str:
        """
        Prepare the output file pattern for feature extraction.

        Args:
            split (str): Data split (e.g., train, test).
            proc (int): Process ID.

        Returns:
            str: Output file pattern.
        """
        file_name = f"{self._cache_file_name}-" if self._cache_file_name else ""
        file_name += f"{split}-"
        file_name += f"{proc:06d}-"
        file_name += "%06d"
        file_name += f"-features-{self._features_key}.msgpack"
        return str(self._dataset_cache_dir / file_name)

    def _features_prepared(self) -> bool:
        """
        Check if features are already prepared.

        Returns:
            bool: True if features are prepared, False otherwise.
        """
        import glob

        from datadings.reader import MsgpackReader

        feature_files = sorted(glob.glob(self._features_path.replace("%06d", "*")))
        if len(feature_files) == 0:
            return False

        total_count = 0
        for feature_file in feature_files:
            try:
                with MsgpackReader(feature_file) as reader:
                    total_count += len(reader)
            except Exception as e:
                logger.warning(
                    f"Feature file {feature_file} is corrupted. Re-running feature extraction. {e}"
                )

                for feature_file in feature_files:
                    Path(feature_file).unlink(missing_ok=True)
                self._features_metadata_path.unlink(missing_ok=True)
                return False

        logger.info(f"Total feature samples found: {total_count}")
        if self._features_metadata_path.exists():
            import pandas as pd

            try:
                metadata = pd.read_csv(self._features_metadata_path)
                assert total_count == len(metadata) and total_count == len(
                    self._dataloader.dataset
                ), (
                    "Features and metadata file lengths do not match. "
                    f"Features: {total_count}, Metadata: {len(metadata)}, Dataset size: {len(self._dataloader.dataset)} "
                    "Re-running feature extraction."
                )
            except AssertionError as e:
                logger.warning(e)
                for feature_file in feature_files:
                    Path(feature_file).unlink(missing_ok=True)
                self._features_metadata_path.unlink(missing_ok=True)
                return False
        else:
            return False
        return True

    def _configure_engine(self, engine: Engine) -> Engine:
        """
        Configure the feature extraction engine.

        Args:
            engine (Engine): Ignite engine instance.
            data_split (str): Data split (e.g., train, test).
        """
        import torch
        from atria_datasets.storage.msgpack_shard_writer import MsgpackShardWriter
        from ignite.engine import Events

        self._configure_test_run(engine=engine)
        self._configure_progress_bar(engine=engine)

        self._features_path_msgpack_writer = MsgpackShardWriter(
            self._features_path, maxcount=self._max_shard_size, overwrite=True
        )

        def write_features(engine: Engine):
            features = engine.state.output
            batch = dict(
                __key__=engine.state.batch["__key__"],
                __index__=engine.state.batch["__index__"],
                **features,
            )

            # convert dict of list to list of dict
            batch = [
                dict(zip(batch, t, strict=True))
                for t in zip(*batch.values(), strict=True)
            ]

            for sample in batch:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.detach().cpu().numpy()
                self._features_path_msgpack_writer.write(sample)
                with open(self._features_metadata_path, "a") as f:
                    f.write(
                        f"{sample['__key__']},{sample['__index__']},{self._features_path_msgpack_writer.fname},{self._features_path_msgpack_writer.count - 1}\n"
                    )

        def cleanup(engine: Engine):
            self._features_path_msgpack_writer.close()

        engine.add_event_handler(Events.ITERATION_COMPLETED, write_features)
        engine.add_event_handler(Events.EPOCH_COMPLETED, cleanup)
        return engine

    def run(self, split: str):
        """
        Run the feature extraction engine.

        Args:
            split (str): Data split (e.g., train, test).

        Returns:
            Engine: Ignite engine instance.
        """
        import ignite.distributed as idist

        self._features_path = self._prepare_output_file_pattern(
            split, proc=idist.get_rank()
        )
        self._features_metadata_path = (
            Path(self._dataset_cache_dir)
            / f"{self._cache_file_name}-{split}-features-{self._features_key}-metadata.csv"
        )

        if self._features_prepared():
            return

        # prepare metadata file
        logger.info(f"Extracting dataset features to: {self._features_path}")
        logger.info(
            f"Saving dataset features metadata to: {self._features_metadata_path}"
        )

        logger.info(f"Preparing features for split: {split}")
        with open(self._features_metadata_path, "w") as f:
            f.write("key,index,features_path,feature_index\n")

        # move task module models to device
        self._model_pipeline.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize engine
        engine = self._initialize_ignite_engine()

        # configure engine
        self._configure_engine(engine, split)

        # run engine
        if self._output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch_size={self._dataloader.batch_size} with output_dir: {self._output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")
        return engine.run(
            self._dataloader,
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )
