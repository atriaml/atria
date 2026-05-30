from http import HTTPStatus
from typing import Any, Optional, Union
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
    after: Union[None, Unset, str] = UNSET,
    pattern: Union[None, Unset, str] = UNSET,
    max_amount: Union[Unset, int] = 100,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_after: Union[None, Unset, str]
    if isinstance(after, Unset):
        json_after = UNSET
    else:
        json_after = after
    params["after"] = json_after

    json_pattern: Union[None, Unset, str]
    if isinstance(pattern, Unset):
        json_pattern = UNSET
    else:
        json_pattern = pattern
    params["pattern"] = json_pattern

    params["max_amount"] = max_amount

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/dataset/{id}/tree/{branch}/{prefix}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]:
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
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]:
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
    after: Union[None, Unset, str] = UNSET,
    pattern: Union[None, Unset, str] = UNSET,
    max_amount: Union[Unset, int] = 100,
) -> Response[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (Union[None, Unset, str]):
        pattern (Union[None, Unset, str]):
        max_amount (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]
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
    after: Union[None, Unset, str] = UNSET,
    pattern: Union[None, Unset, str] = UNSET,
    max_amount: Union[Unset, int] = 100,
) -> Optional[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (Union[None, Unset, str]):
        pattern (Union[None, Unset, str]):
        max_amount (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, LakeFSStoragePaginatedObjects]
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
    after: Union[None, Unset, str] = UNSET,
    pattern: Union[None, Unset, str] = UNSET,
    max_amount: Union[Unset, int] = 100,
) -> Response[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (Union[None, Unset, str]):
        pattern (Union[None, Unset, str]):
        max_amount (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]
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
    after: Union[None, Unset, str] = UNSET,
    pattern: Union[None, Unset, str] = UNSET,
    max_amount: Union[Unset, int] = 100,
) -> Optional[Union[HTTPValidationError, LakeFSStoragePaginatedObjects]]:
    """Tree

    Args:
        id (UUID):
        branch (str):
        prefix (str):
        after (Union[None, Unset, str]):
        pattern (Union[None, Unset, str]):
        max_amount (Union[Unset, int]):  Default: 100.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, LakeFSStoragePaginatedObjects]
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
