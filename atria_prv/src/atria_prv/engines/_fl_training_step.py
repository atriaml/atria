from __future__ import annotations

import random
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps import EngineStep

if TYPE_CHECKING:
    from ignite.engine import Engine

    from atria_prv.trainers._fl_client_trainer import FLClientOutput, FLClientTrainer

logger = get_logger(__name__)


class FLTrainingStep(EngineStep):
    """One federated round = one ignite engine step.

    Self-contained: selects the participating clients for this round (seeded, so the
    selection is reproducible and resume-stable), runs each selected client's local
    update, FedAvg-aggregates their returned parameters into the shared global
    ``model_pipeline`` in place, and returns the round metrics as ``engine.state.output``.

    A DP/secure-aggregation variant subclasses this and overrides ``_aggregate`` only.
    """

    def __init__(
        self,
        model_pipeline,
        device,
        client_trainers: list[FLClientTrainer],
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
        self._client_trainers = client_trainers
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
        """FedAvg over the selected clients: sums each client's returned parameters into
        a running total as it finishes (never holding more than one client's params + the
        running sum at once), then divides by the number of participating clients. Every
        selected client starts from the same global parameter snapshot, taken once before
        the loop, and the average is loaded back into the global model in place."""
        global_params = OrderedDict(
            (k, v.detach().cpu())
            for k, v in self._model_pipeline._model.state_dict().items()
        )

        accumulated: OrderedDict[str, torch.Tensor] | None = None
        last_metrics: dict | None = None
        for client_id in selected_client_ids:
            logger.info(f"[Client {client_id}] update starting")
            output: FLClientOutput = self._client_trainers[client_id].client_update(
                global_params
            )
            if accumulated is None:
                accumulated = output.params
            else:
                for k, v in output.params.items():
                    accumulated[k] += v
            last_metrics = output.metrics
            logger.info(
                f"[Client {client_id}] update finished; metrics: {output.metrics}"
            )
            del output

        averaged = {k: v / len(selected_client_ids) for k, v in accumulated.items()}
        self._model_pipeline._model.load_state_dict(averaged, strict=True)
        logger.info(
            f"Aggregated updates from {len(selected_client_ids)} client(s) into global model"
        )
        return last_metrics

    def __call__(self, engine: Engine, batch: Any) -> dict:
        # rounds map 1:1 to epochs (epoch_length == 1); derive the 0-based round index
        # from engine state so it stays correct across resume (ignite restores state.epoch).
        round_idx = engine.state.epoch - 1
        selected = self._select_clients(round_idx)
        logger.info(
            f"===== [Round {round_idx}/{engine.state.max_epochs - 1}] "
            f"selected clients: {selected} ====="
        )
        metrics = self._aggregate(selected)
        return metrics or {}
