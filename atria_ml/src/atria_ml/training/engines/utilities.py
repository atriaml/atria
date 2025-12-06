from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any

from omegaconf import DictConfig

EXPERIMENT_NAME_KEY = "experiment_name"
METRICS_KEY = "metrics"
TRAINING_ENGINE_KEY = "training_engine"
MODEL_PIPELINE_CHECKPOINT_KEY = "model_pipeline"
CONFIG_KEY = "config"


class FixedBatchIterator:
    def __init__(self, dataloader, fixed_batch_size):
        self.dataloader = dataloader
        self.fixed_batch_size = fixed_batch_size

    def __iter__(self):
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


def _extract_output(x: Any, index: int, key: str) -> Any:
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


def _format_metrics_for_logging(metrics: dict[str, Any]) -> dict[str, Any]:
    import torch

    def convert_for_json(obj) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return str(obj)

    return convert_for_json(metrics)


def _find_differences(dict1: dict | DictConfig, dict2: dict | DictConfig, path=""):
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
