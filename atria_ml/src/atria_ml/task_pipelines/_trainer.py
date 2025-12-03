from atria_core.logger import get_logger
from atria_core.types import TaskType
from atria_datasets.pipelines.atria_data_pipeline import AtriaDataPipeline
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from atria_registry.registry_config import RegistryConfig

from atria_ml.registry import TASK_PIPELINE
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.evaluation import (
    TestEngine,
    ValidationEngine,
    VisualizationEngine,
)
from atria_ml.training.engines.training import TrainingEngine
from atria_ml.training.engines.utilities import RunConfig

logger = get_logger(__name__)

DEFAULTS_SELF = ["_self_"]
DATA_PIPELINE_DEFAULT_LIST = [{"/data_pipeline@data_pipeline": "default"}]
MODEL_DEFAULT_LIST = [{"/model_pipeline@model_pipeline": "???"}]
ENGINE_DEFAULT_LIST = [
    {"/engine@training_engine": "default_training_engine"},
    {"/engine@validation_engine": "default_validation_engine"},
    {"/engine@test_engine": "default_test_engine"},
    {"/engine@visualization_engine": "default_visualization_engine"},
]
OPTIMIZER_DEFAULT_LIST = [
    {"override /optimizer@training_engine.optimizer": "adamw"},
    {"override /lr_scheduler@training_engine.lr_scheduler": "cosine_annealing_lr"},
]


@TASK_PIPELINE.register(
    "trainer",
    configs=[
        RegistryConfig(
            name="default",
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + OPTIMIZER_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + MODEL_DEFAULT_LIST,
            output_dir="outputs/trainer/${resolve_experiment_name:${experiment_name}}",
        ),
        RegistryConfig(
            name=TaskType.image_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + OPTIMIZER_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "image_classification"}],
            output_dir="outputs/trainer/image_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.sequence_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + OPTIMIZER_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "sequence_classification"}],
            output_dir="outputs/trainer/sequence_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.token_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + OPTIMIZER_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "token_classification"}],
            output_dir="outputs/trainer/token_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.layout_token_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + OPTIMIZER_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "layout_token_classification"}],
            output_dir="outputs/trainer/layout_token_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.visual_question_answering.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + OPTIMIZER_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "visual_question_answering"}],
            output_dir="outputs/trainer/visual_question_answering/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.layout_analysis.value,
            defaults=DEFAULTS_SELF
            + DATA_PIPELINE_DEFAULT_LIST
            + ENGINE_DEFAULT_LIST
            + [
                {"/model_pipeline@model_pipeline": "object_detection"},
                {"override /optimizer@training_engine.optimizer": "sgd"},
                {
                    "override /lr_scheduler@training_engine.lr_scheduler": "multi_step_lr"
                },
            ],
            output_dir="outputs/trainer/layout_analysis/${resolve_experiment_name:${experiment_name}}/",
            data_pipeline={"collate_fn": "mmdet_pseudo_collate"},
            training_engine={
                "model_checkpoint_config": {"monitored_metric": None},
                "optimizer": {"lr": 0.02, "weight_decay": 0.0001, "momentum": 0.9},
                "lr_scheduler": {"milestones": [16, 22], "gamma": 0.1},
                "warmup_config": {"warmup_ratio": 0.0, "warmup_steps": 500},
            },
            do_visualization=True,
            visualization_engine={
                "visualize_every_n_epochs": 10,
                "visualize_on_start": False,
            },
            validation_engine={
                "validate_every_n_epochs": 10,
                "validate_on_start": False,
            },
        ),
    ],
    zen_meta={
        "pipeline_name": "atria_trainer",
        "n_devices": 1,
        "backend": "nccl",
        "experiment_name": "_to_be_resolved_",
    },
    zen_exclude=["hydra", "package", "version"],
    is_global_package=True,
)
class Trainer:
    def __init__(
        self,
        data_pipeline: AtriaDataPipeline,
        model_pipeline: AtriaModelPipeline,
        training_engine: TrainingEngine,
        validation_engine: ValidationEngine,
        visualization_engine: VisualizationEngine,
        test_engine: TestEngine,
        output_dir: str,
        resume_checkpoint: str | None = None,
        test_checkpoint: str | None = None,
        seed: int = 42,
        deterministic: bool = False,
        do_train: bool = True,
        do_validation: bool = True,
        do_visualization: bool = False,
        do_test: bool = True,
        vis_batch_size: int = 64,
        test_run: bool = False,
    ):
        self._data_pipeline = data_pipeline
        self._model_pipeline = model_pipeline
        self._training_engine = training_engine if do_train else None
        self._validation_engine = validation_engine if do_validation else None
        self._visualization_engine = visualization_engine if do_visualization else None
        self._test_engine = test_engine if do_test else None
        self._output_dir = output_dir
        self._resume_checkpoint = resume_checkpoint
        self._test_checkpoint = test_checkpoint
        self._seed = seed
        self._deterministic = deterministic
        self._do_train = do_train
        self._do_validation = do_validation
        self._do_visualization = do_visualization
        self._do_test = do_test
        self._visualization_batch_size = vis_batch_size
        self._test_run = test_run

    @property
    def model_pipeline(self):
        return self._model_pipeline

    @property
    def training_engine(self):
        return self._training_engine

    @property
    def validation_engine(self):
        return self._validation_engine

    @property
    def visualization_engine(self):
        return self._visualization_engine

    @property
    def test_engine(self):
        return self._test_engine

    @property
    def device(self):
        return self._device

    @property
    def logger(self):
        return logger

    @property
    def tb_logger(self):
        return self._tb_logger

    def _initialize_runtime(self, local_rank: int) -> None:
        from atria_ml.training.utilities.torch_utils import _initialize_torch

        # initialize training
        logger.info("Initializing torch runtime environment...")
        self._seed = _initialize_torch(
            seed=self._seed, deterministic=self._deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = local_rank

        logger.info(f"Seed set to {self._seed} on device: {self._device}")

    def _setup_logging(self) -> None:
        from atria_ml.training.utilities.torch_utils import _setup_tensorboard

        # initialize logging directory and tensorboard logger
        self._tb_logger = _setup_tensorboard(self._output_dir)

    def _build_data_pipeline(self):
        logger.info("Setting up data pipeline")
        self._data_pipeline.build(
            runtime_transforms=self._model_pipeline.config.runtime_transforms
        )

    def _build_model_pipeline(self) -> None:
        # initialize the task module from partial
        logger.info("Setting up task module")
        self._model_pipeline = self._model_pipeline.build(
            dataset_metadata=self._data_pipeline.dataset_metadata,
            tb_logger=self._tb_logger,
        )
        logger.info(self._model_pipeline)

    def _build_training_dataloaders(self) -> None:
        logger.info(
            "Initializing train dataloader with config: %s",
            self._data_pipeline._dataloader_config.train_config,
        )
        train_dataloader = self._data_pipeline.train_dataloader()
        validation_dataloader = None
        visualization_dataloader = None
        if self._validation_engine is not None:
            # check if validation dataset is available
            logger.info(
                "Initializing validation dataloader with config: %s",
                self._data_pipeline._dataloader_config.eval_config,
            )
            validation_dataloader = self._data_pipeline.validation_dataloader()
            if validation_dataloader is None:
                logger.warning(
                    "You have set do_validation=True but there is no validation dataset available. "
                    "To create a validation dataset from the training dataset, set dataset_splitter in the config. "
                    "Using test dataset for validation."
                )
                validation_dataloader = self._data_pipeline.test_dataloader()
        if self._visualization_engine is not None:
            # initilize the validation engine from partial
            # by default, visualization engine uses the train dataloader as it is
            # generally to be used for generative tasks
            logger.info("Initializing train dataloader for visualization.")
            visualization_dataloader = self._data_pipeline.validation_dataloader(
                batch_size=self._visualization_batch_size
            )
            if visualization_dataloader is None:
                visualization_dataloader = self._data_pipeline.test_dataloader(
                    batch_size=self._visualization_batch_size
                )

        return train_dataloader, validation_dataloader, visualization_dataloader

    def _build_training_engine(self) -> None:
        # initilize the test engine from partial
        train_dataloader, validation_dataloader, visualization_dataloader = (
            self._build_training_dataloaders()
        )
        if self._training_engine is not None:
            logger.info("Setting up training engine")
            self._training_engine: AtriaEngine = self._training_engine.build(
                run_config=self._run_config,
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=train_dataloader,
                device=self._device,
                tb_logger=self._tb_logger,
                validation_engine=self._validation_engine,
                visualization_engine=self._visualization_engine,
            )

        if self._validation_engine is not None:
            logger.info("Setting up validation engine")
            self._validation_engine: AtriaEngine = self._validation_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=validation_dataloader,
                device=self._device,
                tb_logger=self._tb_logger,
            )

        if self._visualization_engine is not None:
            logger.info("Setting up visualization engine")
            self._visualization_engine: AtriaEngine = self._visualization_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=visualization_dataloader,
                device=self._device,
                tb_logger=self._tb_logger,
            )

    def _build_test_engine(self) -> None:
        if self._test_engine is not None:
            logger.info("Setting up test engine")
            logger.info(
                "Initializing train dataloader with config: %s",
                self._data_pipeline._dataloader_config.eval_config,
            )
            test_dataloader = self._data_pipeline.test_dataloader()
            self._test_engine: AtriaEngine = self._test_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=test_dataloader,
                device=self._device,
                tb_logger=self._tb_logger,
            )

    def build(
        self, local_rank: int, experiment_name: str, run_config: RunConfig
    ) -> None:
        import logging

        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(f"Log file path: {handler.baseFilename}")

        self._experiment_name = experiment_name
        self._run_config = run_config
        self._initialize_runtime(local_rank=local_rank)
        self._setup_logging()
        self._build_data_pipeline()
        self._build_model_pipeline()

    def train(self) -> None:
        if self._do_train:
            self._build_training_engine()
            self._training_engine.run(resume_checkpoint=self._resume_checkpoint)

    def test(self) -> None:
        if self._do_test:
            self._build_test_engine()
            self._test_engine.run(test_checkpoint=self._test_checkpoint)

    def run(self) -> None:
        self.train()
        self.test()
