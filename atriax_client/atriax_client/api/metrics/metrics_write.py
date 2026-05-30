from http import HTTPStatus
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.evaluation_metric import EvaluationMetric
from ...models.evaluation_metric_create import EvaluationMetricCreate
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    evaluation_experiment_id: UUID,
    *,
    body: EvaluationMetricCreate | list[EvaluationMetricCreate],
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/evaluation_experiments/{evaluation_experiment_id}/metrics/".format(
            evaluation_experiment_id=quote(str(evaluation_experiment_id), safe=""),
        ),
    }

    if isinstance(body, list):
        _kwargs["json"] = []
        for body_type_0_item_data in body:
            body_type_0_item = body_type_0_item_data.to_dict()
            _kwargs["json"].append(body_type_0_item)

    else:
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> EvaluationMetric | list[EvaluationMetric] | HTTPValidationError | None:
    if response.status_code == 200:

        def _parse_response_200(data: object) -> EvaluationMetric | list[EvaluationMetric]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                response_200_type_0 = []
                _response_200_type_0 = data
                for response_200_type_0_item_data in _response_200_type_0:
                    response_200_type_0_item = EvaluationMetric.from_dict(response_200_type_0_item_data)

                    response_200_type_0.append(response_200_type_0_item)

                return response_200_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_200_type_1 = EvaluationMetric.from_dict(data)

            return response_200_type_1

        response_200 = _parse_response_200(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[EvaluationMetric | list[EvaluationMetric] | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: EvaluationMetricCreate | list[EvaluationMetricCreate],
) -> Response[EvaluationMetric | list[EvaluationMetric] | HTTPValidationError]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (EvaluationMetricCreate | list[EvaluationMetricCreate]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[EvaluationMetric | list[EvaluationMetric] | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: EvaluationMetricCreate | list[EvaluationMetricCreate],
) -> EvaluationMetric | list[EvaluationMetric] | HTTPValidationError | None:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (EvaluationMetricCreate | list[EvaluationMetricCreate]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        EvaluationMetric | list[EvaluationMetric] | HTTPValidationError
    """

    return sync_detailed(
        evaluation_experiment_id=evaluation_experiment_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: EvaluationMetricCreate | list[EvaluationMetricCreate],
) -> Response[EvaluationMetric | list[EvaluationMetric] | HTTPValidationError]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (EvaluationMetricCreate | list[EvaluationMetricCreate]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[EvaluationMetric | list[EvaluationMetric] | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: EvaluationMetricCreate | list[EvaluationMetricCreate],
) -> EvaluationMetric | list[EvaluationMetric] | HTTPValidationError | None:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (EvaluationMetricCreate | list[EvaluationMetricCreate]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        EvaluationMetric | list[EvaluationMetric] | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            client=client,
            body=body,
        )
    ).parsed
