from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.evaluation_metric import EvaluationMetric
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    evaluation_experiment_id: UUID,
    key: int,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/evaluation_experiments/{evaluation_experiment_id}/metrics/{key}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[EvaluationMetric, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = EvaluationMetric.from_dict(response.json())

        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[EvaluationMetric, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    evaluation_experiment_id: UUID,
    key: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[EvaluationMetric, HTTPValidationError]]:
    """Item

    Args:
        evaluation_experiment_id (UUID):
        key (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EvaluationMetric, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        key=key,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    evaluation_experiment_id: UUID,
    key: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[EvaluationMetric, HTTPValidationError]]:
    """Item

    Args:
        evaluation_experiment_id (UUID):
        key (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EvaluationMetric, HTTPValidationError]
    """

    return sync_detailed(
        evaluation_experiment_id=evaluation_experiment_id,
        key=key,
        client=client,
    ).parsed


async def asyncio_detailed(
    evaluation_experiment_id: UUID,
    key: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[EvaluationMetric, HTTPValidationError]]:
    """Item

    Args:
        evaluation_experiment_id (UUID):
        key (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EvaluationMetric, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        key=key,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    evaluation_experiment_id: UUID,
    key: int,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[EvaluationMetric, HTTPValidationError]]:
    """Item

    Args:
        evaluation_experiment_id (UUID):
        key (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EvaluationMetric, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            key=key,
            client=client,
        )
    ).parsed
