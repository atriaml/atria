from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_ml.configs._task import TrainingTaskConfig
from atria_ml.data_pipeline._utilities import auto_dataloader, default_collate
from atria_ml.training.engines._trainer import (
    TrainerEngine,
    TrainerEngineConfig,
    TrainerEngineDependencies,
)

if TYPE_CHECKING:
    import torch
    from atria_datasets.core.dataset._split_iterators import SplitIterator
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
    from ignite.handlers import TensorboardLogger

    from atria_prv.att.configs import MembershipInferenceTaskConfig

logger = get_logger(__name__)

# Fixed checkpoint filename for a trained shadow model, in the same
# ``{"model_pipeline": {"model": ...}}`` format ``ModelPipeline.load_checkpoint`` reads.
SHADOW_CHECKPOINT_NAME = "shadow_model.pt"


class ShadowModelTrainer:
    """Trains (or loads a cached) shadow model on a subsampled ``in`` split.

    Mirrors ``Trainer._build_train_engine`` but trains on an explicit ``in`` ``SplitIterator``
    rather than the full dataset. Each shadow gets its own ``shadow_dir`` so checkpoints do
    not collide; a trained shadow is cached at ``shadow_dir/checkpoints/shadow_model.pt`` and
    reused on subsequent runs when ``shadow_config.reuse_checkpoints`` is set.
    """

    def __init__(
        self,
        config: MembershipInferenceTaskConfig,
        labels,
        device: str | torch.device,
        tb_logger: TensorboardLogger | None = None,
    ) -> None:
        self._config = config
        self._labels = labels
        self._device = device
        self._tb_logger = tb_logger

    def _checkpoint_path(self, shadow_dir: Path) -> Path:
        return shadow_dir / "checkpoints" / SHADOW_CHECKPOINT_NAME

    def _build_model_pipeline(self) -> ModelPipeline:
        return self._config.model_pipeline.build(labels=self._labels)

    def _train_dataloader(self, in_split: SplitIterator):
        import ignite.distributed as idist
        from torch.utils.data import RandomSampler

        return auto_dataloader(
            dataset=in_split,
            collate_fn=default_collate,
            sampler=RandomSampler(in_split),
            drop_last=idist.get_world_size() > 1,
            batch_size=self._config.data.train_batch_size * idist.get_world_size(),
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )

    def _build_engine(
        self, model_pipeline: ModelPipeline, in_split: SplitIterator, shadow_dir: Path
    ) -> TrainerEngine:
        import torch

        trainer_cfg = self._config.shadow_config.trainer
        # We persist the trained shadow ourselves (see ``train``), so disable the ignite
        # checkpointer to avoid depending on its validation-metric monitoring.
        engine_config = TrainerEngineConfig(
            max_epochs=trainer_cfg.max_epochs,
            outputs_to_running_avg=trainer_cfg.outputs_to_running_avg,
            logging=self._config.logging,
            test_run=self._config.test_run,
            with_amp=self._config.with_amp,
            clear_cuda_cache=trainer_cfg.clear_cuda_cache,
            stop_on_nan=trainer_cfg.stop_on_nan,
            eval_training=trainer_cfg.eval_training,
            validate_every_n_epochs=trainer_cfg.validate_every_n_epochs,
            visualize_every_n_epochs=trainer_cfg.visualize_every_n_epochs,
            optimizer=trainer_cfg.optimizer,
            lr_scheduler=trainer_cfg.lr_scheduler,
            model_ema=trainer_cfg.model_ema,
            warmup=trainer_cfg.warmup,
            model_checkpoint=trainer_cfg.model_checkpoint.model_copy(
                update={"enabled": False}
            ),
            gradient=trainer_cfg.gradient,
        )
        # run_config is stored as checkpoint metadata by the engine; a training-shaped
        # view of the attack config satisfies the TrainingTaskConfig type.
        run_config = TrainingTaskConfig(
            env=self._config.env,
            data=self._config.data,
            logging=self._config.logging,
            model_pipeline=self._config.model_pipeline,
            trainer=trainer_cfg,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )
        return TrainerEngine(
            config=engine_config,
            deps=TrainerEngineDependencies(
                model_pipeline=model_pipeline,
                dataloader=self._train_dataloader(in_split),
                device=torch.device(self._device),
                output_dir=str(shadow_dir),
                tb_logger=self._tb_logger,
                run_config=run_config,
            ),
        )

    def train(self, in_split: SplitIterator, shadow_dir: Path) -> ModelPipeline:
        """Return a shadow model trained on ``in_split`` (or loaded from cache)."""
        import torch

        shadow_dir = Path(shadow_dir)
        ckpt_path = self._checkpoint_path(shadow_dir)
        model_pipeline = self._build_model_pipeline()

        if self._config.shadow_config.reuse_checkpoints and ckpt_path.exists():
            logger.info(f"Reusing cached shadow checkpoint: {ckpt_path}")
            model_pipeline.load_checkpoint(str(ckpt_path))
            return model_pipeline

        logger.info(
            f"Training shadow model on {len(in_split)} samples -> {shadow_dir} "
            f"({self._config.shadow_config.trainer.max_epochs} epochs)"
        )
        engine = self._build_engine(model_pipeline, in_split, shadow_dir)
        engine.run()

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_pipeline": model_pipeline.state_dict()}, ckpt_path)
        logger.info(f"Saved shadow checkpoint to {ckpt_path}")
        return model_pipeline
