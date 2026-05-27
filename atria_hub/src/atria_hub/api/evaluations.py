from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from atria_core.logger.logger import get_logger
from atriax_client.models.config_base import ConfigBase
from atriax_client.types import File

from atria_hub.api.base import BaseApi
from atria_hub.client import AtriaHubClient
from atria_hub.exceptions import api_error_handler

if TYPE_CHECKING:
    from atriax_client.models.evaluation_experiment import EvaluationExperiment

logger = get_logger(__name__)


@dataclass
class MetricData:
    name: str
    config: ConfigBase
    data: dict[str, Any]


class SampleExplanationsApi(BaseApi):
    @api_error_handler
    def write(
        self,
        evaluation_experiment_id: UUID,
        name: str,
        config: ConfigBase,
        sample_index: int,
        explanation_metadata: dict[str, Any],
        explanation_payload: bytes,
    ) -> None:
        """Write a sample result to an evaluation."""
        from atriax_client.api.sample_explanations import sample_explanations_write
        from atriax_client.models.body_sample_explanations_write import (
            BodySampleExplanationsWrite,
        )

        with self._client.protected_api_client as client:
            return sample_explanations_write.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                body=BodySampleExplanationsWrite(
                    name=name,
                    sample_index=sample_index,
                    config=json.dumps(config.to_dict()),
                    explanation_metadata=json.dumps(explanation_metadata),
                    explanation_file=File(
                        payload=io.BytesIO(explanation_payload),
                        file_name="explanation.bin",
                        mime_type="application/octet-stream",
                    ),
                ),
            )

    @api_error_handler
    def read(
        self,
        evaluation_experiment_id: UUID,
        sample_index: int,
        config_id: UUID | None = None,
    ) -> list[dict]:
        """Read a batch of sample results from an evaluation."""
        from atriax_client.api.sample_explanations import sample_explanations_read

        with self._client.protected_api_client as client:
            kwargs = {}
            if config_id is not None:
                kwargs["config_id"] = config_id
            return sample_explanations_read.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                sample_index=sample_index,
                **kwargs,
            )


class SampleExplanationMetricsApi(BaseApi):
    def __init__(self, client: AtriaHubClient):
        self._client = client

    @api_error_handler
    def write(
        self,
        evaluation_experiment_id: UUID,
        sample_explanation_id: UUID,
        metric_data: list[MetricData],
    ) -> None:
        """Write a sample result to an evaluation."""
        from atriax_client.api.sample_explanation_metrics import (
            sample_explanation_metrics_write,
        )
        from atriax_client.models.sample_explanation_metric_create import (
            SampleExplanationMetricCreate,
        )
        from atriax_client.models.sample_explanation_metric_create_data import (
            SampleExplanationMetricCreateData,
        )

        with self._client.protected_api_client as client:
            return sample_explanation_metrics_write.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                body=[
                    SampleExplanationMetricCreate(
                        name=d.name,
                        config=d.config,
                        data=SampleExplanationMetricCreateData.from_dict(d.data),
                    )
                    for d in metric_data
                ],
                sample_explanation_id=sample_explanation_id,
            )

    @api_error_handler
    def read(
        self,
        evaluation_experiment_id: UUID,
        sample_index: int,
        sample_explanation_id: UUID | None = None,
        config_id: UUID | None = None,
    ) -> list[dict]:
        """Read a batch of sample results from an evaluation."""
        from atriax_client.api.sample_explanation_metrics import (
            sample_explanation_metrics_read,
        )

        with self._client.protected_api_client as client:
            return sample_explanation_metrics_read.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                sample_explanation_id=sample_explanation_id,
                config_id=config_id,
                sample_index=sample_index,
            )


class SampleEvaluationApi(BaseApi):
    @api_error_handler
    def list_indices(self, evaluation_experiment_id: UUID) -> list[int]:
        """Read a batch of sample results from an evaluation."""
        from atriax_client.api.sample_evaluations import sample_evaluations_list_indices

        with self._client.protected_api_client as client:
            return sample_evaluations_list_indices.sync_detailed(
                client=client, evaluation_experiment_id=evaluation_experiment_id
            )

    @api_error_handler
    def write(
        self, evaluation_experiment_id: UUID, data_per_sample_index: dict[int, dict]
    ) -> None:
        """Write a sample result to an evaluation."""
        from atriax_client.api.sample_evaluations import sample_evaluations_write
        from atriax_client.models import (
            SampleEvaluationCreate,
            SampleEvaluationCreateData,
        )

        with self._client.protected_api_client as client:
            return sample_evaluations_write.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                body=[
                    SampleEvaluationCreate(
                        sample_index=sample_index,
                        data=SampleEvaluationCreateData.from_dict(data),
                    )
                    for sample_index, data in data_per_sample_index.items()
                ],
            )

    @api_error_handler
    def read(
        self, evaluation_experiment_id: UUID, sample_indices: list[int]
    ) -> list[dict]:
        """Read a batch of sample results from an evaluation."""
        from atriax_client.api.sample_evaluations import sample_evaluations_read

        with self._client.protected_api_client as client:
            return sample_evaluations_read.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                body=sample_indices,
            )


class EvaluationMetricsApi(BaseApi):
    def __init__(self, client: AtriaHubClient):
        self._client = client

    @api_error_handler
    def write(self, evaluation_experiment_id: UUID, metrics: dict[str, Any]) -> None:
        """Write a sample result to an evaluation."""
        from atriax_client.api.metrics import metrics_write
        from atriax_client.models.evaluation_metric_create import EvaluationMetricCreate

        logger.info("metrics %s", metrics)
        with self._client.protected_api_client as client:
            return metrics_write.sync_detailed(
                client=client,
                evaluation_experiment_id=evaluation_experiment_id,
                body=[
                    EvaluationMetricCreate(key=key, value=value)
                    for key, value in metrics.items()
                ],
            )

    @api_error_handler
    def read(self, evaluation_experiment_id: UUID) -> list[dict]:
        """Read a batch of sample results from an evaluation."""
        from atriax_client.api.metrics import metrics_read

        with self._client.protected_api_client as client:
            return metrics_read.sync_detailed(
                client=client, evaluation_experiment_id=evaluation_experiment_id
            )


class EvaluationsApi(BaseApi):
    def __init__(self, client: AtriaHubClient):
        self._client = client
        self._sample_evaluations = SampleEvaluationApi(client)
        self._metrics = EvaluationMetricsApi(client)
        self._sample_explanations = SampleExplanationsApi(client)
        self._sample_explanation_metrics = SampleExplanationMetricsApi(client)

    @property
    def sample_evaluations(self) -> SampleEvaluationApi:
        """Access to sample results API."""
        return self._sample_evaluations

    @property
    def metrics(self) -> EvaluationMetricsApi:
        """Access to metrics API."""
        return self._metrics

    @property
    def sample_explanations(self) -> SampleExplanationsApi:
        """Access to sample explanations API."""
        return self._sample_explanations

    @property
    def sample_explanation_metrics(self) -> SampleExplanationMetricsApi:
        """Access to sample explanation metrics API."""
        return self._sample_explanation_metrics

    @api_error_handler
    def get_or_create(
        self,
        dataset_id: UUID,
        dataset_branch: str,
        dataset_config_name: str,
        dataset_split: str,
        model_id: UUID,
        model_branch: str,
        model_config_name: str,
    ) -> EvaluationExperiment:
        """Retrieve a dataset from the hub by its name."""

        from atriax_client.api.evaluation_experiments import (
            evaluation_experiments_get_or_create,
        )
        from atriax_client.models.evaluation_experiment_get_or_create import (
            EvaluationExperimentGetOrCreate,
        )

        with self._client.protected_api_client as client:
            return evaluation_experiments_get_or_create.sync_detailed(
                client=client,
                body=EvaluationExperimentGetOrCreate(
                    dataset_id=dataset_id,
                    dataset_branch=dataset_branch,
                    dataset_config_name=dataset_config_name,
                    dataset_split=dataset_split,
                    model_id=model_id,
                    model_branch=model_branch,
                    model_config_name=model_config_name,
                ),
            )

    @api_error_handler
    def delete(self, id: uuid.UUID) -> None:
        from atriax_client.api.dataset import dataset_delete

        with self._client.protected_api_client as client:
            return dataset_delete.sync_detailed(client=client, id=id)
