from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.tracking_get_tracking_metrics_response_200_item import TrackingGetTrackingMetricsResponse200Item
from ...types import UNSET, Response


def _get_kwargs(
    *,
    experiment_id: str,
    sample_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["experiment_id"] = experiment_id

    params["sample_id"] = sample_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/tracking/metrics",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TrackingGetTrackingMetricsResponse200Item.from_dict(response_200_item_data)

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
) -> Response[HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    experiment_id: str,
    sample_id: str,
) -> Response[HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]]:
    """Get Tracking Metrics

     Fetch attrs.json for all metric runs that have data for the given sample.

    Args:
        experiment_id (str):
        sample_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]]
    """

    kwargs = _get_kwargs(
        experiment_id=experiment_id,
        sample_id=sample_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    experiment_id: str,
    sample_id: str,
) -> HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item] | None:
    """Get Tracking Metrics

     Fetch attrs.json for all metric runs that have data for the given sample.

    Args:
        experiment_id (str):
        sample_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]
    """

    return sync_detailed(
        client=client,
        experiment_id=experiment_id,
        sample_id=sample_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    experiment_id: str,
    sample_id: str,
) -> Response[HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]]:
    """Get Tracking Metrics

     Fetch attrs.json for all metric runs that have data for the given sample.

    Args:
        experiment_id (str):
        sample_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]]
    """

    kwargs = _get_kwargs(
        experiment_id=experiment_id,
        sample_id=sample_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    experiment_id: str,
    sample_id: str,
) -> HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item] | None:
    """Get Tracking Metrics

     Fetch attrs.json for all metric runs that have data for the given sample.

    Args:
        experiment_id (str):
        sample_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[TrackingGetTrackingMetricsResponse200Item]
    """

    return (
        await asyncio_detailed(
            client=client,
            experiment_id=experiment_id,
            sample_id=sample_id,
        )
    ).parsed
