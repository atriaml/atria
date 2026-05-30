from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.config import Config
from ...models.config_type import ConfigType
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    page: int | Unset = 0,
    page_size: int | Unset = 100,
    paginated: bool | Unset = True,
    order_by: str | Unset = "created_at",
    order: str | Unset = "desc",
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["page"] = page

    params["page_size"] = page_size

    params["paginated"] = paginated

    params["order_by"] = order_by

    params["order"] = order

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

    json_config_type: None | str | Unset
    if isinstance(config_type, Unset):
        json_config_type = UNSET
    elif isinstance(config_type, ConfigType):
        json_config_type = config_type.value
    else:
        json_config_type = config_type
    params["config_type"] = json_config_type

    json_name: None | str | Unset
    if isinstance(name, Unset):
        json_name = UNSET
    else:
        json_name = name
    params["name"] = json_name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/configs/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[Config] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Config.from_dict(response_200_item_data)

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
) -> Response[HTTPValidationError | list[Config]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    page: int | Unset = 0,
    page_size: int | Unset = 100,
    paginated: bool | Unset = True,
    order_by: str | Unset = "created_at",
    order: str | Unset = "desc",
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | list[Config]]:
    """List

    Args:
        page (int | Unset):  Default: 0.
        page_size (int | Unset):  Default: 100.
        paginated (bool | Unset):  Default: True.
        order_by (str | Unset):  Default: 'created_at'.
        order (str | Unset):  Default: 'desc'.
        search (None | str | Unset):
        search_by (None | str | Unset):
        config_type (ConfigType | None | Unset):
        name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[Config]]
    """

    kwargs = _get_kwargs(
        page=page,
        page_size=page_size,
        paginated=paginated,
        order_by=order_by,
        order=order,
        search=search,
        search_by=search_by,
        config_type=config_type,
        name=name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    page: int | Unset = 0,
    page_size: int | Unset = 100,
    paginated: bool | Unset = True,
    order_by: str | Unset = "created_at",
    order: str | Unset = "desc",
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | str | Unset = UNSET,
) -> HTTPValidationError | list[Config] | None:
    """List

    Args:
        page (int | Unset):  Default: 0.
        page_size (int | Unset):  Default: 100.
        paginated (bool | Unset):  Default: True.
        order_by (str | Unset):  Default: 'created_at'.
        order (str | Unset):  Default: 'desc'.
        search (None | str | Unset):
        search_by (None | str | Unset):
        config_type (ConfigType | None | Unset):
        name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[Config]
    """

    return sync_detailed(
        client=client,
        page=page,
        page_size=page_size,
        paginated=paginated,
        order_by=order_by,
        order=order,
        search=search,
        search_by=search_by,
        config_type=config_type,
        name=name,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    page: int | Unset = 0,
    page_size: int | Unset = 100,
    paginated: bool | Unset = True,
    order_by: str | Unset = "created_at",
    order: str | Unset = "desc",
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | list[Config]]:
    """List

    Args:
        page (int | Unset):  Default: 0.
        page_size (int | Unset):  Default: 100.
        paginated (bool | Unset):  Default: True.
        order_by (str | Unset):  Default: 'created_at'.
        order (str | Unset):  Default: 'desc'.
        search (None | str | Unset):
        search_by (None | str | Unset):
        config_type (ConfigType | None | Unset):
        name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[Config]]
    """

    kwargs = _get_kwargs(
        page=page,
        page_size=page_size,
        paginated=paginated,
        order_by=order_by,
        order=order,
        search=search,
        search_by=search_by,
        config_type=config_type,
        name=name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    page: int | Unset = 0,
    page_size: int | Unset = 100,
    paginated: bool | Unset = True,
    order_by: str | Unset = "created_at",
    order: str | Unset = "desc",
    search: None | str | Unset = UNSET,
    search_by: None | str | Unset = UNSET,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | str | Unset = UNSET,
) -> HTTPValidationError | list[Config] | None:
    """List

    Args:
        page (int | Unset):  Default: 0.
        page_size (int | Unset):  Default: 100.
        paginated (bool | Unset):  Default: True.
        order_by (str | Unset):  Default: 'created_at'.
        order (str | Unset):  Default: 'desc'.
        search (None | str | Unset):
        search_by (None | str | Unset):
        config_type (ConfigType | None | Unset):
        name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[Config]
    """

    return (
        await asyncio_detailed(
            client=client,
            page=page,
            page_size=page_size,
            paginated=paginated,
            order_by=order_by,
            order=order,
            search=search,
            search_by=search_by,
            config_type=config_type,
            name=name,
        )
    ).parsed
