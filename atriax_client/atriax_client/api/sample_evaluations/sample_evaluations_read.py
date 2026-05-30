from http import HTTPStatus
from typing import Any
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sample_evaluation import SampleEvaluation
from ...types import Response


def _get_kwargs(
    evaluation_experiment_id: UUID,
    *,
    body: list[int],
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/evaluation_experiments/{evaluation_experiment_id}/sample_evaluations/",
    }

    _kwargs["json"] = body

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list["SampleEvaluation"] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = SampleEvaluation.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[HTTPValidationError | list["SampleEvaluation"]]:
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
    body: list[int],
) -> Response[HTTPValidationError | list["SampleEvaluation"]]:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['SampleEvaluation']]]
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
    body: list[int],
) -> HTTPValidationError | list["SampleEvaluation"] | None:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['SampleEvaluation']]
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
    body: list[int],
) -> Response[HTTPValidationError | list["SampleEvaluation"]]:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['SampleEvaluation']]]
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
    body: list[int],
) -> HTTPValidationError | list["SampleEvaluation"] | None:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['SampleEvaluation']]
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            client=client,
            body=body,
        )
    ).parsed
