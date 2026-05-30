from http import HTTPStatus
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.lake_fs_storage_paginated_objects import LakeFSStoragePaginatedObjects
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: UUID,
    branch: str,
    prefix: str,
    *,
    after: None | str | Unset = UNSET,
    pattern: None | str | Unset = UNSET,
    max_amount: int | Unset = 100,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_after: None | str | Unset
    if isinstance(after, Unset):
        json_after = UNSET
    else:
        json_after = after
    params["after"] = json_after

    json_pattern: None | str | Unset
    if isinstance(pattern, Unset):
        json_pattern = UNSET
    else:
        json_pattern = pattern
    params["pattern"] = json_pattern

    params["max_amount"] = max_amount

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/model/{id}/tree/{branch}/{prefix}/".format(
            id=quote(str(id), safe=""),
            branch=quote(str(branch), safe=""),
            prefix=quote(str(prefix), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | LakeFSStoragePaginatedObjects | None:
    if response.status_code == 200:
        response_200 = LakeFSStoragePaginatedObjects.from_dict(response.json())

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
) -> Response[HTTPValidationError | LakeFSStoragePaginatedObjects]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: UUID,
    branch: str,
    prefix: str,
    *,
    client: AuthenticatedClient,
    after: None | str | Unset = UNSET,
    pattern: None | str | Unset = UNSET,
    max_amount: int | Unset = 100,
) -> Response[HTTPValidationError | LakeFSStoragePaginatedObjects]:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (None | str | Unset):
        pattern (None | str | Unset):
        max_amount (int | Unset):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | LakeFSStoragePaginatedObjects]
    """

    kwargs = _get_kwargs(
        id=id,
        branch=branch,
        prefix=prefix,
        after=after,
        pattern=pattern,
        max_amount=max_amount,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: UUID,
    branch: str,
    prefix: str,
    *,
    client: AuthenticatedClient,
    after: None | str | Unset = UNSET,
    pattern: None | str | Unset = UNSET,
    max_amount: int | Unset = 100,
) -> HTTPValidationError | LakeFSStoragePaginatedObjects | None:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (None | str | Unset):
        pattern (None | str | Unset):
        max_amount (int | Unset):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | LakeFSStoragePaginatedObjects
    """

    return sync_detailed(
        id=id,
        branch=branch,
        prefix=prefix,
        client=client,
        after=after,
        pattern=pattern,
        max_amount=max_amount,
    ).parsed


async def asyncio_detailed(
    id: UUID,
    branch: str,
    prefix: str,
    *,
    client: AuthenticatedClient,
    after: None | str | Unset = UNSET,
    pattern: None | str | Unset = UNSET,
    max_amount: int | Unset = 100,
) -> Response[HTTPValidationError | LakeFSStoragePaginatedObjects]:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (None | str | Unset):
        pattern (None | str | Unset):
        max_amount (int | Unset):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | LakeFSStoragePaginatedObjects]
    """

    kwargs = _get_kwargs(
        id=id,
        branch=branch,
        prefix=prefix,
        after=after,
        pattern=pattern,
        max_amount=max_amount,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: UUID,
    branch: str,
    prefix: str,
    *,
    client: AuthenticatedClient,
    after: None | str | Unset = UNSET,
    pattern: None | str | Unset = UNSET,
    max_amount: int | Unset = 100,
) -> HTTPValidationError | LakeFSStoragePaginatedObjects | None:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (None | str | Unset):
        pattern (None | str | Unset):
        max_amount (int | Unset):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | LakeFSStoragePaginatedObjects
    """

    return (
        await asyncio_detailed(
            id=id,
            branch=branch,
            prefix=prefix,
            client=client,
            after=after,
            pattern=pattern,
            max_amount=max_amount,
        )
    ).parsed
