from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: UUID,
    branch: str,
    split: str,
    *,
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_search: None | str | Unset
    if isinstance(search, Unset):
        json_search = UNSET
    else:
        json_search = search
    params["search"] = json_search

    json_search_by: None | str | Unset
    if isinstance(search_by, Unset):
        json_search_by = UNSET
    else:
        json_search_by = search_by
    params["search_by"] = json_search_by

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/dataset/{id}/ids/{branch}/{split}/".format(
            id=quote(str(id), safe=""),
            branch=quote(str(branch), safe=""),
            split=quote(str(split), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[str] | None:
    if response.status_code == 200:
        response_200 = cast(list[str], response.json())

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
) -> Response[HTTPValidationError | list[str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: UUID,
    branch: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | list[str]]:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        split (str):
        search (None | str | Unset):
        search_by (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[str]]
    """

    kwargs = _get_kwargs(
        id=id,
        branch=branch,
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
    split: str,
    *,
    client: AuthenticatedClient,
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
) -> HTTPValidationError | list[str] | None:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        split (str):
        search (None | str | Unset):
        search_by (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[str]
    """

    return sync_detailed(
        id=id,
        branch=branch,
        split=split,
        client=client,
        search=search,
        search_by=search_by,
    ).parsed


async def asyncio_detailed(
    id: UUID,
    branch: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | list[str]]:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        split (str):
        search (None | str | Unset):
        search_by (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[str]]
    """

    kwargs = _get_kwargs(
        id=id,
        branch=branch,
        split=split,
        search=search,
        search_by=search_by,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: UUID,
    branch: str,
    split: str,
    *,
    client: AuthenticatedClient,
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
) -> HTTPValidationError | list[str] | None:
    """Get All Sample Ids

    Args:
        id (UUID):
        branch (str):
        split (str):
        search (None | str | Unset):
        search_by (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[str]
    """

    return (
        await asyncio_detailed(
            id=id,
            branch=branch,
            split=split,
            client=client,
            search=search,
            search_by=search_by,
        )
    ).parsed
