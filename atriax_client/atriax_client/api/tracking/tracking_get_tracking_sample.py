from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.tracking_get_tracking_sample_response_tracking_get_tracking_sample import (
    TrackingGetTrackingSampleResponseTrackingGetTrackingSample,
)
from ...types import UNSET, Response


def _get_kwargs(
    sample_id: str,
    *,
    run_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["run_id"] = run_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/tracking/samples/{sample_id}".format(
            sample_id=quote(str(sample_id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample | None:
    if response.status_code == 200:
        response_200 = TrackingGetTrackingSampleResponseTrackingGetTrackingSample.from_dict(response.json())

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
) -> Response[HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    sample_id: str,
    *,
    client: AuthenticatedClient,
    run_id: str,
) -> Response[HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample]:
    """Get Tracking Sample

    Args:
        sample_id (str):
        run_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample]
    """

    kwargs = _get_kwargs(
        sample_id=sample_id,
        run_id=run_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    sample_id: str,
    *,
    client: AuthenticatedClient,
    run_id: str,
) -> HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample | None:
    """Get Tracking Sample

    Args:
        sample_id (str):
        run_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample
    """

    return sync_detailed(
        sample_id=sample_id,
        client=client,
        run_id=run_id,
    ).parsed


async def asyncio_detailed(
    sample_id: str,
    *,
    client: AuthenticatedClient,
    run_id: str,
) -> Response[HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample]:
    """Get Tracking Sample

    Args:
        sample_id (str):
        run_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample]
    """

    kwargs = _get_kwargs(
        sample_id=sample_id,
        run_id=run_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    sample_id: str,
    *,
    client: AuthenticatedClient,
    run_id: str,
) -> HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample | None:
    """Get Tracking Sample

    Args:
        sample_id (str):
        run_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TrackingGetTrackingSampleResponseTrackingGetTrackingSample
    """

    return (
        await asyncio_detailed(
            sample_id=sample_id,
            client=client,
            run_id=run_id,
        )
    ).parsed
