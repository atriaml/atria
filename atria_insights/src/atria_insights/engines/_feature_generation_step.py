from collections import OrderedDict
from pathlib import Path

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine

from atria_insights.data_types._features import BatchFeatures
from atria_insights.model_pipelines._model_pipeline import (
    _DEFAULT_FEATURE_INPUT_KEY,
    ExplainableModelPipeline,
)
from atria_insights.storage.sample_cache_managers._features_cacher import FeaturesCacher

logger = get_logger(__name__)


class FeatureGenerationStep(EngineStep):
    def __init__(
        self,
        x_model_pipeline: ExplainableModelPipeline,
        device: str | torch.device,
        with_amp: bool = False,
        cache_dir: str | Path | None = None,
        file_name: str = "features.hdf5",
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=x_model_pipeline._model_pipeline,
            device=device,
            with_amp=with_amp,
            test_run=test_run,
        )

        self._x_model_pipeline = x_model_pipeline

        assert cache_dir is not None, "cache_dir must be provided if caching is enabled"
        self._cacher = FeaturesCacher(cache_dir=cache_dir, file_name=file_name)

    @property
    def name(self) -> str:
        return "baselines"

    def _get_batch_features(
        self, engine: Engine, batch: list[TensorDataModel]
    ) -> BatchFeatures:
        self._x_model_pipeline.ops.eval()
        collated_batch = batch[0].batch(batch).ops.to(self._device)
        with torch.no_grad():
            # prepare explained inputs
            inputs = self._x_model_pipeline._explained_inputs(batch=collated_batch)
            if isinstance(inputs, OrderedDict):
                input_feature_keys = tuple(inputs.keys())
                assert len(inputs) == len(input_feature_keys), (
                    "Input feature keys length does not match inputs length."
                    f" {len(input_feature_keys)=}, {len(inputs)=}"
                )
                features = tuple(inputs.values())
            else:
                input_feature_keys = (_DEFAULT_FEATURE_INPUT_KEY,)
                features = (inputs,)
        return BatchFeatures(
            sample_id=collated_batch.metadata.sample_id,
            feature_keys=input_feature_keys,
            features=features,
        )

    def _save_to_disk(self, batch_features: BatchFeatures) -> None:
        # conver to list
        for sample_features in batch_features.tolist():
            sample_id = sample_features.sample_id
            self._cacher.save_sample(data=sample_features)
            logger.debug(f"Saved explanation to cache for sample_id: {sample_id}")

    def __call__(self, engine: Engine, batch: list[TensorDataModel]) -> BatchFeatures:
        """Process batch with optional caching."""
        batch_features = self._get_batch_features(engine, batch)
        self._save_to_disk(batch_features)
        return batch_features
