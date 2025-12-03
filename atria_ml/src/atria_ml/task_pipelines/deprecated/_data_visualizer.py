from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger
from atria_core.types import DatasetSplitType
from atria_datasets.pipelines.atria_data_pipeline import AtriaDataPipeline

from atria_ml.registry import TASK_PIPELINE

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


@TASK_PIPELINE.register(
    "data_visualizer",
    defaults=["_self_", {"/data_pipeline@data_pipeline": "default"}],
    zen_meta={
        "hydra": {
            "run": {"dir": "/tmp/data_visualizer"},
            "output_subdir": "hydra",
            "job": {"chdir": False},
            "searchpath": ["pkg://atria/conf", "pkg://atria_examples/conf"],
        }
    },
    is_global_package=True,
)
class DataVisualizer:
    def __init__(
        self,
        data_pipeline: AtriaDataPipeline,
        seed: int = 42,
        deterministic: bool = False,
        backend: str | None = "nccl",
        n_devices: int = 1,
        split: DatasetSplitType | None = None,
    ):
        self._data_pipeline = data_pipeline
        self._seed = seed
        self._deterministic = deterministic
        self._backend = backend
        self._n_devices = n_devices
        self._split = split
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    # ---------------------------
    # Properties
    # ---------------------------

    @property
    def train_dataloader(self) -> "DataLoader | None":
        return self._train_dataloader

    @property
    def val_dataloader(self) -> "DataLoader | None":
        return self._val_dataloader

    @property
    def test_dataloader(self) -> "DataLoader | None":
        return self._test_dataloader

    # ---------------------------
    # Public Methods
    # ---------------------------

    def build(self) -> None:
        """Builds the data pipeline and initializes torch."""
        from atria_ml.training.utilities.torch_utils import _initialize_torch

        logger.info("Building DataVisualizer...")
        _initialize_torch(seed=self._seed, deterministic=self._deterministic)
        self._data_pipeline.build(split=self._split)
        self._train_dataloader = self._data_pipeline.train_dataloader()
        self._val_dataloader = self._data_pipeline.validation_dataloader()
        self._test_dataloader = self._data_pipeline.test_dataloader()

    def run(self) -> None:
        logger.info("Running DataVisualizer...")
        self._visualize_split("Training", self.train_dataloader)
        self._visualize_split("Validation", self.val_dataloader)
        self._visualize_split("Testing", self.test_dataloader)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        from rich.pretty import pretty_repr

        return pretty_repr(
            {
                "class": self.__class__.__name__,
                "config": {
                    "seed": self._seed,
                    "deterministic": self._deterministic,
                    "backend": self._backend,
                    "n_devices": self._n_devices,
                    "split": self._split,
                },
                "dataloaders": {
                    "train": (
                        self.train_dataloader
                        if self.train_dataloader is not None
                        else None
                    ),
                    "val": (
                        self.val_dataloader if self.val_dataloader is not None else None
                    ),
                    "test": (
                        self.test_dataloader
                        if self.test_dataloader is not None
                        else None
                    ),
                },
            }
        )

    # ---------------------------
    # Private Methods
    # ---------------------------

    def _visualize_split(self, name: str, dataloader: "DataLoader | None") -> None:
        if dataloader is not None:
            logger.info(f"Showing {name} batch...")
            for batch in dataloader:
                logger.info(batch)
                break
