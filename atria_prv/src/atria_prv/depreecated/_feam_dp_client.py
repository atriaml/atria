from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from atria_logger import get_logger

from atria_prv.trainers._fl_client import FLClient, FLClientOutput
from atria_prv.opacus.privacy_engine import _PrivacyEngine

if TYPE_CHECKING:
    from atria_prv.configs import DPConfig

logger = get_logger(__name__)


class FeAmDPClient(FLClient):
    def __init__(self, *args, dp_config: DPConfig, sigma_k: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dp_config = dp_config
        self._sigma_k = sigma_k

        from opacus.validators import ModuleValidator  # same pattern as existing DPTrainer._build()

        if not ModuleValidator.is_valid(self._model_pipeline._model):
            self._model_pipeline._model = ModuleValidator.fix(self._model_pipeline._model)

    def client_update(
        self, global_params: OrderedDict[str, torch.Tensor]
    ) -> FLClientOutput:
        """Overrides FLClient.client_update entirely (not just _build_train_engine --
        unused here, no internal TrainerEngine needed for a single step) because the
        return semantics also differ: FLClient returns post-step WEIGHTS (full
        state_dict); FeAmDPClient returns a pseudo-GRADIENT (before-after delta, over
        trainable params only)."""
        self._model_pipeline._model.load_state_dict(global_params, strict=True)
        self._model_pipeline.ops.to_device(self._device)
        self._model_pipeline._model.train()

        # DP batch size is derived from the sample rate on this client's local shard so
        # the Opacus-derived Poisson rate (batch_size / len(shard)) equals
        # dp_config.sample_rate -- NOT the fixed data.train_batch_size (which would give
        # an uncontrolled rate). Since all K equal-sized clients participate every round,
        # this rate matches the intended sample rate on the original full data.
        num_train_samples = len(self._data_pipeline.dataset.train)
        batch_size = max(1, math.ceil(num_train_samples * self._dp_config.sample_rate))
        dataloader = self._data_pipeline.train_dataloader(
            batch_size=batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        sgd_opt = torch.optim.SGD(
            self._model_pipeline._model.parameters(), lr=1.0, momentum=0.0
        )
        # fresh every round -- genuinely throwaway, no accountant state needs to persist
        privacy_engine = _PrivacyEngine(accountant=self._dp_config.accountant)
        hooks, dp_opt, dp_dataloader = privacy_engine.make_private(
            module=self._model_pipeline._model,
            optimizer=sgd_opt,
            data_loader=dataloader,
            noise_multiplier=self._sigma_k,
            max_grad_norm=self._dp_config.max_grad_norm,
            poisson_sampling=True,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        batch = next(iter(dp_dataloader))
        loss = None
        dp_opt.zero_grad()
        if len(batch) > 0:
            collated_batch = batch[0].batch(batch).ops.to_torch().ops.to(self._device)
            model_output = self._model_pipeline.training_step(batch=collated_batch)
            model_output.loss.backward()
            dp_opt.step()
            loss = model_output.loss.item()
        # else: empty Poisson-sampled batch this round (rare, small shards) -- skip the
        # step entirely; this client's pseudo-gradient is then all-zero this round.

        hooks.cleanup()  # removes opacus's forward/backward hooks + monkey-patched param
                          # attrs (grad_sample, _forward_counter, ...). Required: since
                          # self._model_pipeline persists across rounds, skipping this
                          # would stack hooks from every prior round the client
                          # participated in, corrupting gradients over time.

        self._model_pipeline.ops.to_device(torch.device("cpu"))
        pseudo_grad = OrderedDict(
            (name, (global_params[name].cpu() - p.detach().cpu()))
            for name, p in self._model_pipeline._model.named_parameters()
        )  # lr=1 SGD: param_new = param_old - g~  =>  g~ = param_old - param_new.
           # Only over named_parameters() (trainable) -- NOT full state_dict() (buffers
           # aren't gradients; not an issue for the BERT-based model this repo trains,
           # which has no such buffers -- accepted simplification, not fixed).
        return FLClientOutput(
            params=pseudo_grad, metrics={"loss": loss} if loss is not None else None
        )
