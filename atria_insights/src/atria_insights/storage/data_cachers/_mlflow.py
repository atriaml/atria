from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atria_logger import get_logger

from atria_insights.storage.data_cachers._common import SerializableSampleData

logger = get_logger(__name__)


def _flatten_tensors(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Recursively flatten a nested dict of tensors into {flat_key: ndarray}.

    Nesting separator is ``__``.
    """
    result: dict[str, np.ndarray] = {}
    if isinstance(obj, torch.Tensor):
        result[prefix] = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        result[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child_key = f"{prefix}__{k}" if prefix else k
            result.update(_flatten_tensors(v, child_key))
    else:
        raise ValueError(f"Unsupported tensor type {type(obj)} for key '{prefix}'")
    return result


def _unflatten_tensors(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reconstruct nested dict of torch.Tensors from flat npz dict."""
    nested: dict[str, Any] = {}
    for flat_key, array in flat.items():
        parts = flat_key.split("__")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = torch.from_numpy(array)
    return nested


class MLflowDataCacher:
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        artifact_prefix: str,
        tracking_uri: str | None = None,
    ) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        self._experiment_name = experiment_name
        self._run_name = run_name
        self._artifact_prefix = artifact_prefix

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._client = MlflowClient()

        if run_name:
            self._run_id = run_name
        else:
            self._run_id = self._get_or_create_run(experiment_name, run_name)

    @property
    def run_id(self) -> str:
        return self._run_id

    def save_file_attrs(self, attrs: dict[str, Any]) -> None:
        data = {k: v for k, v in attrs.items() if v is not None}
        self._client.log_dict(
            self._run_id, data, f"{self._artifact_prefix}/_run_metadata.json"
        )

    def sample_exists(self, sample_key: str) -> bool:
        safe_key = self._safe_key(sample_key)
        prefix = f"{self._artifact_prefix}/{safe_key}"
        artifacts = self._client.list_artifacts(self._run_id, path=prefix)
        return len(artifacts) > 0

    def list_sample_keys(self) -> list[str]:
        artifacts = self._client.list_artifacts(
            self._run_id, path=self._artifact_prefix
        )
        keys = []
        for a in artifacts:
            # each entry is a directory named after the (safe) sample key
            if a.is_dir:
                keys.append(Path(a.path).name)
        return keys

    def save_sample(self, data: SerializableSampleData) -> None:
        safe_key = self._safe_key(data.sample_id)
        base = f"{self._artifact_prefix}/{safe_key}"

        # attrs → JSON via MlflowClient.log_dict (no active run required)
        meta: dict[str, Any] = {"sample_id": data.sample_id}
        if data.attrs:
            meta.update(data.attrs)
        self._client.log_dict(self._run_id, meta, f"{base}/metadata.json")

        # tensors → npz bytes via MlflowClient.log_stream
        if data.tensors:
            flat = {}
            for k, v in data.tensors.items():
                if v is None:
                    continue
                flat.update(_flatten_tensors(v, prefix=k))
            buf = io.BytesIO()
            np.savez(buf, **flat)
            buf.seek(0)
            self._client.log_stream(self._run_id, buf, f"{base}/tensors.npz")

        logger.debug(
            "Saved sample '%s' to run '%s' at '%s'.", data.sample_id, self._run_id, base
        )

    def load_sample(
        self, sample_key: str, load_tensors: bool = True
    ) -> SerializableSampleData:
        import mlflow

        safe_key = self._safe_key(sample_key)
        base = f"{self._artifact_prefix}/{safe_key}"

        meta_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{self._run_id}/{base}/metadata.json"
        )
        with open(meta_path) as f:
            raw_attrs = json.load(f)

        # Keep sample_id in attrs too — HDF5DataCacher does the same
        stored_id = raw_attrs.get("sample_id", sample_key)
        attrs = raw_attrs

        tensors = None
        if load_tensors:
            # tensors file may not exist (e.g. summary cacher stores no tensors)
            try:
                npz_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{self._run_id}/{base}/tensors.npz"
                )
                flat = dict(np.load(npz_path, allow_pickle=False))
                tensors = _unflatten_tensors(flat)
            except Exception:
                tensors = None

        return SerializableSampleData(sample_id=stored_id, attrs=attrs, tensors=tensors)

    def load_sample_attrs(self, sample_key: str) -> dict[str, Any]:
        import mlflow

        safe_key = self._safe_key(sample_key)
        meta_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{self._run_id}/{self._artifact_prefix}/{safe_key}/metadata.json"
        )
        with open(meta_path) as f:
            raw = json.load(f)
        raw.pop("sample_id", None)
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_key(sample_key: str) -> str:
        """Replace characters that are invalid in artifact paths."""
        return sample_key.replace("/", "_").replace("\\", "_")

    def _get_or_create_run(self, experiment_name: str, run_name: str) -> str:
        import mlflow

        experiment = mlflow.set_experiment(experiment_name)
        exp_id = experiment.experiment_id

        existing = self._client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        )
        if existing:
            run_id = existing[0].info.run_id
            logger.debug("Reusing existing MLflow run '%s' (id=%s).", run_name, run_id)
            return run_id

        run = self._client.create_run(experiment_id=exp_id, run_name=run_name)
        logger.debug("Created new MLflow run '%s' (id=%s).", run_name, run.info.run_id)
        return run.info.run_id
