from http import HTTPStatus
from typing import Any
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sample_explanation_metric import SampleExplanationMetric
from ...types import UNSET, Response, Unset


def _get_kwargs(
    evaluation_experiment_id: UUID,
    *,
    sample_explanation_id: None | UUID | Unset = UNSET,
    sample_index: None | Unset | list[int] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_sample_explanation_id: None | Unset | str
    if isinstance(sample_explanation_id, Unset):
        json_sample_explanation_id = UNSET
    elif isinstance(sample_explanation_id, UUID):
        json_sample_explanation_id = str(sample_explanation_id)
    else:
        json_sample_explanation_id = sample_explanation_id
    params["sample_explanation_id"] = json_sample_explanation_id

    json_sample_index: None | Unset | list[int]
    if isinstance(sample_index, Unset):
        json_sample_index = UNSET
    elif isinstance(sample_index, list):
        json_sample_index = sample_index

    else:
        json_sample_index = sample_index
    params["sample_index"] = json_sample_index

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/evaluations/{evaluation_experiment_id}/sample_explanation_metrics/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list["SampleExplanationMetric"] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = SampleExplanationMetric.from_dict(response_200_item_data)

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
) -> Response[HTTPValidationError | list["SampleExplanationMetric"]]:
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
    sample_explanation_id: None | UUID | Unset = UNSET,
    sample_index: None | Unset | list[int] = UNSET,
) -> Response[HTTPValidationError | list["SampleExplanationMetric"]]:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (Union[None, UUID, Unset]):
        sample_index (Union[None, Unset, list[int]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['SampleExplanationMetric']]]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        sample_explanation_id=sample_explanation_id,
        sample_index=sample_index,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    sample_explanation_id: None | UUID | Unset = UNSET,
    sample_index: None | Unset | list[int] = UNSET,
) -> HTTPValidationError | list["SampleExplanationMetric"] | None:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (Union[None, UUID, Unset]):
        sample_index (Union[None, Unset, list[int]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['SampleExplanationMetric']]
    """

    return sync_detailed(
        evaluation_experiment_id=evaluation_experiment_id,
        client=client,
        sample_explanation_id=sample_explanation_id,
        sample_index=sample_index,
    ).parsed


async def asyncio_detailed(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    sample_explanation_id: None | UUID | Unset = UNSET,
    sample_index: None | Unset | list[int] = UNSET,
) -> Response[HTTPValidationError | list["SampleExplanationMetric"]]:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (Union[None, UUID, Unset]):
        sample_index (Union[None, Unset, list[int]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['SampleExplanationMetric']]]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        sample_explanation_id=sample_explanation_id,
        sample_index=sample_index,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    sample_explanation_id: None | UUID | Unset = UNSET,
    sample_index: None | Unset | list[int] = UNSET,
) -> HTTPValidationError | list["SampleExplanationMetric"] | None:
    """Read

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (Union[None, UUID, Unset]):
        sample_index (Union[None, Unset, list[int]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['SampleExplanationMetric']]
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            client=client,
            sample_explanation_id=sample_explanation_id,
            sample_index=sample_index,
        )
    ).parsed
