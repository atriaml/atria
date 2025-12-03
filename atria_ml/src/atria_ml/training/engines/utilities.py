"""
Engine Utilities Module

This module provides utility functions and classes for managing and configuring
training, validation, and testing engines in the Atria framework. These utilities
include functions for attaching metrics, logging outputs, handling GPU stats, and
more.

Classes:
    - RunConfig: Class for managing and comparing run configurations.
    - FixedBatchIterator: Iterator for fixed-size batches.

Functions:
    - _empty_cuda_cache: Clears the CUDA cache.
    - _extract_output: Extracts specific output from a mapping or sequence.
    - _detach_tensors: Detaches tensors from the computation graph.
    - _convert_tensor_to_half: Converts tensors to half precision.
    - _log_training_metrics: Logs training metrics.
    - _log_eval_metrics: Logs evaluation metrics.
    - _attach_metrics_to_engine: Attaches metrics to an engine.
    - _attach_nan_callback_to_engine: Attaches a callback to handle NaN values.
    - _attach_cuda_cache_callback_to_engine: Attaches a callback to clear CUDA cache.
    - _attach_gpu_stats_callback_to_engine: Attaches a callback to log GPU stats.
    - _attach_time_profiler_to_engine: Attaches a time profiler to the engine.
    - _attach_output_logging_to_engine: Attaches output logging to the engine.
    - _print_optimizers_info: Prints information about configured optimizers.
    - _print_schedulers_info: Prints information about configured learning rate schedulers.
    - _find_differences: Finds differences between two dictionaries.

Dependencies:
    - ignite: For distributed training and engine management.
    - torch: For tensor operations and model handling.
    - omegaconf: For configuration management.
    - sqlalchemy: For database engine handling.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from atria_core.logger.logger import get_logger
from atria_core.types import ModelOutput
from atria_core.utilities.strings import _indent_string
from omegaconf import DictConfig

from atria_ml.training.utilities.constants import TrainingStage

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine
    from ignite.metrics import Metric

logger = get_logger(__name__)

EXPERIMENT_NAME_KEY = "experiment_name"
METRICS_KEY = "metrics"
TRAINING_ENGINE_KEY = "training_engine"
MODEL_PIPELINE_CHECKPOINT_KEY = "model_pipeline"
RUN_CONFIG_KEY = "run_config"


class RunConfig:
    """
    Class for managing and comparing run configurations.

    Attributes:
        _data (Dict): The configuration data.

    Methods:
        state_dict: Returns the configuration data as a dictionary.
        compare_configs: Compares the current configuration with a target configuration.
        load_state_dict: Loads configuration data from a state dictionary.
    """

    def __init__(self, data: DictConfig):
        """
        Initializes the RunConfig object.

        Args:
            data (DictConfig): The configuration data.
        """
        self._data = data

    @property
    def data(self) -> DictConfig:
        """
        Returns the configuration data.

        Returns:
            DictConfig: The configuration data.
        """
        return self._data

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the configuration data as a dictionary.

        Returns:
            Dict[str, Any]: The configuration data.
        """
        return self._data

    def compare_configs(self, target_data: dict) -> bool:
        """
        Compares the current configuration with a target configuration.

        Args:
            target_data (dict): The target configuration data.

        Returns:
            bool: True if configurations are identical, False otherwise.
        """
        differences = _find_differences(self._data, target_data)
        if len(differences) > 0:
            logger.warning(
                "You are trying to continue a training run with different configuration from the previous run."
            )
            logger.warning("Differences:")
            for diff in differences:
                logger.warning(
                    f"Key: {diff[0]}, Previous value: {diff[2]}, New value: {diff[1]}"
                )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads configuration data from a state dictionary.

        Args:
            state_dict (Dict[str, Any]): The state dictionary containing configuration data.
        """


class InferenceConfig(RunConfig):
    pass


class FixedBatchIterator:
    """
    Iterator for fixed-size batches.

    Attributes:
        dataloader: The data loader providing batches.
        fixed_batch_size: The fixed size of batches.

    Methods:
        __iter__: Iterates over the data loader and yields fixed-size batches.
    """

    def __init__(self, dataloader, fixed_batch_size):
        """
        Initializes the FixedBatchIterator object.

        Args:
            dataloader: The data loader providing batches.
            fixed_batch_size (int): The fixed size of batches.
        """
        self.dataloader = dataloader
        self.fixed_batch_size = fixed_batch_size

    def __iter__(self):
        """
        Iterates over the data loader and yields fixed-size batches.

        Yields:
            dict: A batch of data with fixed size.
        """
        import torch

        total_samples = 0
        current_batch = None
        for batch in self.dataloader:
            total_samples += batch["label"].shape[0]
            if current_batch is None:
                current_batch = {k: v for k, v in batch.items()}  # noqa: C416
            else:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        current_batch[k] = torch.cat([current_batch[k], v], dim=0)
                    else:
                        current_batch[k].extend(v)
            while len(current_batch["__key__"]) >= self.fixed_batch_size:
                yielded_batch = {
                    k: v[: self.fixed_batch_size] for k, v in current_batch.items()
                }
                yield yielded_batch
                current_batch = {
                    k: v[self.fixed_batch_size :] for k, v in current_batch.items()
                }
        if current_batch:
            yield current_batch


def _empty_cuda_cache(_) -> None:
    """
    Clears the CUDA cache to free up memory.
    """
    import torch

    torch.cuda.empty_cache()
    import gc

    gc.collect()


def _extract_output(x: Any, index: int, key: str) -> Any:
    """
    Extracts specific output from a mapping or sequence.

    Args:
        x (Any): The input data.
        index (int): The index to extract from a sequence.
        key (str): The key to extract from a mapping.

    Returns:
        Any: The extracted output.
    """
    import numbers

    import pydantic
    import torch

    if isinstance(x, Mapping):
        return x[key]
    elif isinstance(x, Sequence):
        return x[index]
    elif isinstance(x, torch.Tensor | numbers.Number):
        return x
    elif is_dataclass(x):
        return getattr(x, key)
    elif isinstance(x, pydantic.BaseModel):
        return getattr(x, key)
    else:
        raise TypeError(
            "Unhandled type of update_function's output. "
            f"It should either mapping or sequence, but given {type(x)}"
        )


def _detach_tensors(
    input: dict[str, "torch.Tensor"] | ModelOutput,
) -> dict[str, "torch.Tensor"] | ModelOutput:
    """
    Detaches tensors from the computation graph.

    Args:
        input (Union[Dict[str, Tensor], ModelOutput]): The input data.

    Returns:
        Union[Dict[str, Tensor], ModelOutput]: The detached data.
    """
    from ignite.utils import apply_to_tensor

    if isinstance(input, ModelOutput):
        import torch

        for field_name, field_value in input.__dict__.items():
            if isinstance(field_value, torch.Tensor):
                setattr(input, field_name, field_value.detach())
    else:
        return apply_to_tensor(input, lambda input: input.detach())


def _convert_tensor_to_half(
    model_output: dict[str, "torch.Tensor"] | ModelOutput,
) -> dict[str, "torch.Tensor"] | ModelOutput:
    """
    Converts tensors to half precision.

    Args:
        model_output (Union[Dict[str, Tensor], ModelOutput]): The model output.

    Returns:
        Union[Dict[str, Tensor], ModelOutput]: The converted output.
    """
    from ignite.utils import apply_to_tensor

    # detach all the outputs from the graph
    return apply_to_tensor(model_output, lambda tensor: tensor.half())


def _log_training_metrics(logger, epoch, elapsed, tag, metrics):
    """
    Logs training metrics.

    Args:
        logger: The logger object.
        epoch (int): The current epoch.
        elapsed (float): The elapsed time.
        tag (str): The tag for the metrics.
        metrics (dict): The metrics to log.
    """
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Training time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def _log_eval_metrics(logger, epoch, elapsed, tag, metrics):
    """
    Logs evaluation metrics.

    Args:
        logger: The logger object.
        epoch (int): The current epoch.
        elapsed (float): The elapsed time.
        tag (str): The tag for the metrics.
        metrics (dict): The metrics to log.
    """
    logger.info(
        "Epoch %d - Evaluation time: %.2fs - %s metrics: EpochResult:",
        epoch,
        elapsed,
        tag,
    )
    for k, v in metrics.items():
        logger.info(f"\t{k}: {v}")


def _attach_metrics_to_engine(
    engine: "Engine",
    metrics: dict[str, "Metric"],
    stage: TrainingStage,
    prefix: str = None,
):
    """
    Attaches metrics to an engine.

    Args:
        engine (Engine): The engine object.
        metrics (Dict[str, Metric]): The metrics to attach.
        stage (TrainingStage): The training stage.
        prefix (str, optional): The prefix for metric names.
    """
    from ignite.metrics.metric import EpochWise

    for metric_name, metric in metrics.items():
        logger.debug(f"Attaching metric {metric_name} to engine")
        metric.attach(
            engine,
            (
                f"{stage}/{metric_name}"
                if prefix is None
                else f"{prefix}/{stage}/{metric_name}"
            ),
            usage=EpochWise(),
        )


def _attach_nan_callback_to_engine(engine: "Engine"):
    """
    Attaches a callback to handle NaN values.

    Args:
        engine (Engine): The engine object.
    """
    from ignite.engine import Events

    from atria_ml.training.handlers.terminate_on_nan import TerminateOnNan

    engine.add_event_handler(
        Events.ITERATION_COMPLETED,
        TerminateOnNan(output_transform=lambda x: x.__dict__),
    )


def _attach_cuda_cache_callback_to_engine(engine: "Engine"):
    """
    Attaches a callback to clear CUDA cache.

    Args:
        engine (Engine): The engine object.
    """
    import torch
    from ignite.engine import Events

    if torch.cuda.is_available():
        engine.add_event_handler(Events.EPOCH_COMPLETED, _empty_cuda_cache)


def _attach_gpu_stats_callback_to_engine(engine: "Engine", logging_steps: int):
    """
    Attaches a callback to log GPU stats.

    Args:
        engine (Engine): The engine object.
        logging_steps (int): The number of steps between logging.
    """
    import ignite.distributed as idist
    import torch

    if idist.device() != torch.device("cpu"):
        from ignite.contrib.metrics import GpuInfo
        from ignite.engine import Events

        GpuInfo().attach(
            engine,
            name="gpu",
            event_name=Events.ITERATION_COMPLETED(every=logging_steps),
        )


def _attach_time_profiler_to_engine(engine: "Engine"):
    """
    Attaches a time profiler to the engine.

    Args:
        engine (Engine): The engine object.
    """
    from ignite.engine import Events
    from ignite.handlers import BasicTimeProfiler, HandlersTimeProfiler

    handlers_profiler = HandlersTimeProfiler()
    basic_printer = BasicTimeProfiler()
    basic_printer.attach(engine)
    handlers_profiler.attach(engine)

    @engine.on(Events.EPOCH_COMPLETED)
    def log_intermediate_results():
        basic_printer.print_results(basic_printer.get_results())
        handlers_profiler.print_results(handlers_profiler.get_results())


def _attach_output_logging_to_engine(
    engine: "Engine",
    stage: TrainingStage,
    outputs_to_running_avg: list[str],
    alpha=0.95,
):
    """
    Attaches output logging to the engine.

    Args:
        engine (Engine): The engine object.
        stage (TrainingStage): The training stage.
        outputs_to_running_avg (List[str]): The outputs to log as running averages.
        alpha (float, optional): The smoothing factor for running averages.
    """
    from ignite.metrics import RunningAverage

    for index, key in enumerate(outputs_to_running_avg):
        RunningAverage(
            alpha=alpha,
            output_transform=partial(_extract_output, index=index, key=key),
            epoch_bound=True,
        ).attach(engine, f"{stage}/running_avg_{key}")


def _print_optimizers_info(optimizers: dict[str, "torch.optim.Optimizer"]) -> None:
    """
    Prints information about configured optimizers.

    Args:
        optimizers (Dict[str, torch.optim.Optimizer]): The optimizers to print.
    """
    # print information
    msg = "Configured optimizers:\n"
    for k, v in optimizers.items():
        opt_str = _indent_string(str(v))
        msg += f"{k}:\n"
        msg += f"{opt_str}\n"
    logger.info(msg)


def _print_schedulers_info(
    lr_schedulers: dict[str, "torch.optim.lr_scheduler.LRScheduler"],
) -> None:
    """
    Prints information about configured learning rate schedulers.

    Args:
        lr_schedulers (Dict[str, torch.optim.lr_scheduler.LRScheduler]): The schedulers to print.
    """
    if lr_schedulers is not None:
        msg = "Configured learning rate schedulers: \n"
        for k, v in lr_schedulers.items():
            lr_sch_str = _indent_string(str(v))
            msg += f"{k}:\n"
            msg += f"{lr_sch_str}\n"
        logger.info(msg)


def _find_differences(dict1: dict | DictConfig, dict2: dict | DictConfig, path=""):
    """
    Finds differences between two dictionaries.

    Args:
        dict1 (Union[dict, DictConfig]): The first dictionary.
        dict2 (Union[dict, DictConfig]): The second dictionary.
        path (str, optional): The current path in the dictionary hierarchy.

    Returns:
        List[Tuple[str, Any, Any]]: A list of differences.
    """
    differences = []
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    for key in keys1 - keys2:
        differences.append((f"{path}.{key}".strip("."), dict1[key], None))
    for key in keys2 - keys1:
        differences.append((f"{path}.{key}".strip("."), None, dict2[key]))
    for key in keys1 & keys2:
        value1 = dict1[key]
        value2 = dict2[key]
        current_path = f"{path}.{key}".strip(".")

        if isinstance(value1, dict | DictConfig) and isinstance(
            value2, dict | DictConfig
        ):
            differences.extend(_find_differences(value1, value2, current_path))
        elif value1 != value2:
            differences.append((current_path, value1, value2))

    return differences
