from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from atria_logger import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME_KEY = "experiment_name"
METRICS_KEY = "metrics"
TRAINING_ENGINE_KEY = "training_engine"
MODEL_PIPELINE_CHECKPOINT_KEY = "model_pipeline"
RUN_CONFIG_KEY = "run_config"


def _reset_random_seeds(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _initialize_torch(seed: int = 0, deterministic: bool = False):
    _reset_random_seeds(seed)

    # Configure CuDNN backend for deterministic behavior if required
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _get_env_info():
    import platform
    import sys

    import ignite.distributed as idist
    import psutil

    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": list(platform.architecture()),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "pytorch_version": str(torch.__version__),
    }

    # Memory information
    memory = psutil.virtual_memory()
    system_info["total_ram_gb"] = memory.total / 1024**3
    system_info["available_ram_gb"] = memory.available / 1024**3
    system_info["ram_usage_percent"] = memory.percent

    # GPU information
    if torch.cuda.is_available():
        system_info["world_size"] = idist.get_world_size()
        system_info["rank"] = idist.get_rank()
        system_info["cuda_available"] = True
        system_info["cuda_version"] = torch.version.cuda
        system_info["gpu_count"] = torch.cuda.device_count()
        system_info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            system_info["gpus"].append(
                {
                    "id": i,
                    "name": gpu_props.name,
                    "total_memory_gb": gpu_props.total_memory / 1024**3,
                }
            )
    else:
        system_info["cuda_available"] = False

    return system_info


def _find_checkpoint(output_dir: str | Path, checkpoint_type: str) -> str | None:
    import glob
    import os

    checkpoint_dir = Path(output_dir) / "checkpoints"

    if not checkpoint_dir.exists():
        return None

    available_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")
    if checkpoint_type == "last":
        # get the latest checkpoint following pattern checkpoint_n.pt
        available_checkpoints = [
            c
            for c in available_checkpoints
            if os.path.basename(c).startswith("checkpoint_")
        ]

        # sort by epoch number
        available_checkpoints.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
            reverse=True,
        )

        return available_checkpoints[0] if len(available_checkpoints) > 0 else None
    elif checkpoint_type == "best":
        # get the best checkpoint following pattern best_checkpoint.pt
        available_checkpoints = [
            c
            for c in available_checkpoints
            if os.path.basename(c).startswith("best_checkpoint")
        ]

        # sort best checkpoints by epoch number
        available_checkpoints.sort(
            key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]),
            reverse=True,
        )

        return available_checkpoints[0] if len(available_checkpoints) > 0 else None
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
