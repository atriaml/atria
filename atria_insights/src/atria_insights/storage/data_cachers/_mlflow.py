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
        self._run_id = self._get_or_create_run(experiment_name, run_name)
        self._artifact_repo = self._build_artifact_repo()

    @property
    def run_id(self) -> str:
        return self._run_id

    def save_file_attrs(self, attrs: dict[str, Any]) -> None:
        for key, data in attrs.items():
            self._client.log_dict(self._run_id, data, f"{key}.json")

    def sample_exists(self, sample_key: str) -> bool:
        safe_key = self._safe_key(sample_key)
        prefix = f"{self._artifact_prefix}/{safe_key}"
        return len(self._artifact_repo.list_artifacts(prefix)) > 0

    def list_sample_keys(self) -> list[str]:
        artifacts = self._artifact_repo.list_artifacts(self._artifact_prefix)
        return [Path(a.path).name for a in artifacts if a.is_dir]

    def save_sample(self, data: SerializableSampleData) -> None:
        safe_key = self._safe_key(data.sample_id)
        base = f"{self._artifact_prefix}/{safe_key}"

        # Attrs → attrs.json (human-readable, supports nested dicts)
        attrs: dict[str, Any] = {"sample_id": data.sample_id}
        if data.attrs:
            attrs.update(data.attrs)
        attrs_buf = io.BytesIO(json.dumps(attrs).encode("utf-8"))
        self._client.log_stream(self._run_id, attrs_buf, f"{base}/attrs.json")

        # Tensors → tensors.npz
        if data.tensors:
            flat: dict[str, np.ndarray] = {}
            for k, v in data.tensors.items():
                if v is not None:
                    flat.update(_flatten_tensors(v, prefix=k))
            if flat:
                buf = io.BytesIO()
                np.savez(buf, **flat)
                buf.seek(0)
                self._client.log_stream(self._run_id, buf, f"{base}/tensors.npz")

        logger.debug(
            "Saved sample '%s' to run '%s' at '%s'.", data.sample_id, self._run_id, base
        )

    def _read_artifact_json(self, artifact_path: str) -> dict[str, Any]:
        import mlflow

        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{self._run_id}/{artifact_path}"
        )
        with open(local_path) as f:
            return json.load(f)

    def _read_artifact_bytes(self, artifact_path: str) -> io.BytesIO:
        import mlflow

        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{self._run_id}/{artifact_path}"
        )
        with open(local_path, "rb") as f:
            return io.BytesIO(f.read())

    def load_sample(
        self, sample_key: str, load_tensors: bool = True
    ) -> SerializableSampleData:
        safe_key = self._safe_key(sample_key)
        base = f"{self._artifact_prefix}/{safe_key}"
        attrs = self._read_artifact_json(f"{base}/attrs.json")
        stored_id = attrs.get("sample_id", sample_key)
        tensors = None
        if load_tensors:
            tensors_path = f"{base}/tensors.npz"
            artifacts = self._artifact_repo.list_artifacts(base)
            has_tensors = any(a.path.endswith("tensors.npz") for a in artifacts)
            if has_tensors:
                try:
                    buf = self._read_artifact_bytes(tensors_path)
                    flat = dict(np.load(buf, allow_pickle=False))
                    tensors = _unflatten_tensors(flat) if flat else None
                except Exception:
                    logger.warning(
                        "Failed to load tensors for sample '%s', skipping.", sample_key
                    )
            else:
                logger.debug(
                    "No tensors.npz found for sample '%s', skipping tensor load.",
                    sample_key,
                )
        return SerializableSampleData(sample_id=stored_id, attrs=attrs, tensors=tensors)

    def load_sample_attrs(self, sample_key: str) -> dict[str, Any]:
        safe_key = self._safe_key(sample_key)
        base = f"{self._artifact_prefix}/{safe_key}"

        attrs = self._read_artifact_json(f"{base}/attrs.json")
        attrs.pop("sample_id", None)
        return attrs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_artifact_repo(self):
        from mlflow.store.artifact.artifact_repository_registry import (
            get_artifact_repository,
        )

        artifact_uri = self._client.get_run(self._run_id).info.artifact_uri
        return get_artifact_repository(artifact_uri)

    @staticmethod
    def _safe_key(sample_key: str) -> str:
        """Sanitise a sample key for use as an artifact path.

        Forward slashes are intentional path separators (e.g. metric_key/sample_id)
        and are preserved.  Only backslashes are replaced.
        """
        return sample_key.replace("\\", "_")

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
