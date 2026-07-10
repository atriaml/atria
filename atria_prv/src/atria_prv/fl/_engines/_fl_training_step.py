from __future__ import annotations

import gc
import hashlib
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps import EngineStep

from atria_prv.fl._trainers._fl_client_trainer import FLClientOutput, FLClientTrainer
from atria_prv.fl.configs import FLClientTrainingTaskConfig

if TYPE_CHECKING:
    from ignite.engine import Engine

logger = get_logger(__name__)


def hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()

    for name in sorted(state_dict.keys()):
        h.update(name.encode())

        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())

    return h.hexdigest()


class FLTrainingStep(EngineStep):
    def __init__(
        self,
        model_pipeline,
        device,
        client_training_task_configs: list[FLClientTrainingTaskConfig],
        total_num_clients: int,
        client_fraction: float,
        seed: int,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=model_pipeline,
            device=device,
            with_amp=with_amp,
            test_run=test_run,
        )
        self._client_training_task_configs = client_training_task_configs
        self._total_num_clients = total_num_clients
        self._client_fraction = client_fraction
        self._seed = seed
        self._num_clients_per_round = max(1, round(client_fraction * total_num_clients))

    @property
    def name(self) -> str:
        return "fl_training"

    def _select_clients(self, round_idx: int) -> list[int]:
        rng = random.Random(f"{self._seed}:{round_idx}")
        return sorted(
            rng.sample(range(self._total_num_clients), self._num_clients_per_round)
        )

    def _aggregate(self, selected_client_ids: list[int]) -> dict | None:
        global_params = OrderedDict(
            (k, v.detach().cpu())
            for k, v in self._model_pipeline._model.state_dict().items()
        )

        accumulated: OrderedDict[str, torch.Tensor] | None = None
        total_samples = 0
        last_metrics: dict | None = None
        for client_id in selected_client_ids:
            logger.info(f"[Client {client_id}] update starting")
            # initialize the client trainer with the config for this client, which includes its data partition
            trainer = FLClientTrainer(
                config=self._client_training_task_configs[client_id],
                model_pipeline=self._model_pipeline,
            )
            output: FLClientOutput = trainer.train(global_params)
            # weight each client's update by its local sample count (FedAvg)
            n = output.num_samples
            total_samples += n
            if accumulated is None:
                accumulated = OrderedDict(
                    (k, v * n) for k, v in output.params.items()
                )
            else:
                for k, v in output.params.items():
                    accumulated[k] += v * n
            last_metrics = output.metrics
            logger.info(
                f"[Client {client_id}] update finished; "
                f"num_samples: {n}; metrics: {output.metrics}"
            )
            del output
            del trainer

            # remove gpu cache
            with torch.no_grad():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        assert (
            total_samples > 0
        ), "Cannot aggregate: selected clients contributed zero samples"
        averaged = {k: v / total_samples for k, v in accumulated.items()}
        self._model_pipeline._model.load_state_dict(averaged, strict=True)
        logger.info(
            f"Aggregated updates from {len(selected_client_ids)} client(s) / "
            f"{total_samples} samples into global model"
        )
        return last_metrics

    def __call__(self, engine: Engine, batch: Any) -> dict:
        round_idx = engine.state.epoch - 1
        selected = self._select_clients(round_idx)
        logger.info(
            f"===== [Round {round_idx}/{engine.state.max_epochs - 1}] "
            f"selected clients: {selected} ====="
        )
        metrics = self._aggregate(selected)
        return metrics or {}
