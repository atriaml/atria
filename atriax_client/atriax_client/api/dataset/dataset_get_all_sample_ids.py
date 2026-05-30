from http import HTTPStatus
from typing import Any, Optional, Union, cast
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: UUID,
    branch: str,
    config: str,
    split: str,
    *,
    search: Union[None, Unset, str] = UNSET,
    search_by: Union[None, Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_search: Union[None, Unset, str]
    if isinstance(search, Unset):
        json_search = UNSET
    else:
        json_search = search
    params["search"] = json_search

    json_search_by: Union[None, Unset, str]
    if isinstance(search_by, Unset):
        json_search_by = UNSET
    else:
        json_search_by = search_by
    params["search_by"] = json_search_by

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/dataset/{id}/ids/{branch}/{config}/{split}/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list[int]]]:
    if response.status_code == 200:
        response_200 = cast(list[int], response.json())

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
) -> Response[Union[HTTPValidationError, list[int]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: UUID,
    branch: str,
    config: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: Union[None, Unset, str] = UNSET,
    search_by: Union[None, Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list[int]]]:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        config (str):
        split (str):
        search (Union[None, Unset, str]):
        search_by (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list[int]]]
    """

    kwargs = _get_kwargs(
        id=id,
        branch=branch,
        config=config,
        split=split,
        search=search,
        search_by=search_by,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: UUID,
    branch: str,
    config: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: Union[None, Unset, str] = UNSET,
    search_by: Union[None, Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list[int]]]:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        config (str):
        split (str):
        search (Union[None, Unset, str]):
        search_by (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list[int]]
    """

    return sync_detailed(
        id=id,
        branch=branch,
        config=config,
        split=split,
        client=client,
        search=search,
        search_by=search_by,
    ).parsed


async def asyncio_detailed(
    id: UUID,
    branch: str,
    config: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: Union[None, Unset, str] = UNSET,
    search_by: Union[None, Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, list[int]]]:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        config (str):
        split (str):
        search (Union[None, Unset, str]):
        search_by (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list[int]]]
    """

    kwargs = _get_kwargs(
        id=id,
        branch=branch,
        config=config,
        split=split,
        search=search,
        search_by=search_by,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: UUID,
    branch: str,
    config: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: Union[None, Unset, str] = UNSET,
    search_by: Union[None, Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, list[int]]]:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        config (str):
        split (str):
        search (Union[None, Unset, str]):
        search_by (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list[int]]
    """

    return (
        await asyncio_detailed(
            id=id,
            branch=branch,
            config=config,
            split=split,
            client=client,
            search=search,
            search_by=search_by,
        )
    ).parsed
