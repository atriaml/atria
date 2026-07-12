from __future__ import annotations

import gc
import hashlib
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps import EngineStep

from atria_prv.fl._engines._fl_aggregation import FLAggregationStrategy
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


class BaseFLTrainingStep(EngineStep):
    """Common FL round flow: select clients, run each locally, aggregate.

    Subclasses decide how a client's ``FLClientTrainer`` is obtained and released
    (fresh every round vs. cached across rounds); the aggregation math itself lives
    in the injected :class:`FLAggregationStrategy`.
    """

    def __init__(
        self,
        model_pipeline,
        device,
        client_training_task_configs: list[FLClientTrainingTaskConfig],
        total_num_clients: int,
        client_fraction: float,
        seed: int,
        aggregation: FLAggregationStrategy,
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
        self._aggregation = aggregation

    def _acquire_trainer(self, client_id: int) -> FLClientTrainer:
        """Return the client's trainer for this round. Overridden by subclasses."""
        raise NotImplementedError

    def _release_trainer(self, client_id: int, trainer: FLClientTrainer) -> None:
        """Release the client's trainer after its update. Overridden by subclasses."""
        raise NotImplementedError

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

        self._aggregation.reset()
        last_metrics: dict | None = None
        for client_id in selected_client_ids:
            logger.info(f"[Client {client_id}] update starting")
            trainer = self._acquire_trainer(client_id)
            output: FLClientOutput = trainer.train(global_params)

            # # for sanity check lets print first few values of first 10 params of the model
            # for idx, (name, param) in enumerate(output.params.items()):
            #     if idx >= 10:
            #         break
            #     logger.info(
            #         f"[Client {client_id}] model param {name}: "
            #         f"{param.detach().cpu().numpy().flatten()[:10]}"
            #     )

            self._aggregation.update(output)
            last_metrics = output.metrics
            logger.info(
                f"[Client {client_id}] update finished; "
                f"num_samples: {output.num_samples}; metrics: {output.metrics}"
            )
            del output
            self._release_trainer(client_id, trainer)
            del trainer

            # remove gpu cache
            with torch.no_grad():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        averaged = self._aggregation.compute()
        self._model_pipeline._model.load_state_dict(averaged, strict=True)
        logger.info(
            f"Aggregated updates from {len(selected_client_ids)} client(s) "
            f"into global model"
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


class FLTrainingStep(BaseFLTrainingStep):
    """Builds a fresh ``FLClientTrainer`` (and its data pipeline) every round."""

    def _acquire_trainer(self, client_id: int) -> FLClientTrainer:
        # initialize a fresh trainer for this client, which includes its data partition
        return FLClientTrainer(
            config=self._client_training_task_configs[client_id],
            model_pipeline=self._model_pipeline,
        )

    def _release_trainer(self, client_id: int, trainer: FLClientTrainer) -> None:
        # nothing to retain; the caller's `del trainer` frees it
        pass


class CachedFLTrainingStep(BaseFLTrainingStep):
    """Keeps each client's ``FLClientTrainer`` alive across rounds.

    Safe because the model pipeline is shared (passed directly, not replicated per
    client); trades higher resident memory for avoiding per-round dataset/data
    pipeline rebuilds.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client_trainers: dict[int, FLClientTrainer] = {}

    def _acquire_trainer(self, client_id: int) -> FLClientTrainer:
        trainer = self._client_trainers.get(client_id)
        if trainer is None:
            trainer = FLClientTrainer(
                config=self._client_training_task_configs[client_id],
                model_pipeline=self._model_pipeline,
            )
            self._client_trainers[client_id] = trainer
        return trainer

    def _release_trainer(self, client_id: int, trainer: FLClientTrainer) -> None:
        # keep the trainer (and its built dataset/data pipeline) for the next round
        pass
