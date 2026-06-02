from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.config_schemas_list_config_schemas_response_200_item import ConfigSchemasListConfigSchemasResponse200Item
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    group_name: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_group_name: None | str | Unset
    if isinstance(group_name, Unset):
        json_group_name = UNSET
    else:
        json_group_name = group_name
    params["group_name"] = json_group_name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/config_schemas/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = ConfigSchemasListConfigSchemasResponse200Item.from_dict(response_200_item_data)

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
) -> Response[HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    group_name: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]]:
    """List Config Schemas

    Args:
        group_name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]]
    """

    kwargs = _get_kwargs(
        group_name=group_name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    group_name: None | str | Unset = UNSET,
) -> HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item] | None:
    """List Config Schemas

    Args:
        group_name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]
    """

    return sync_detailed(
        client=client,
        group_name=group_name,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    group_name: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]]:
    """List Config Schemas

    Args:
        group_name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]]
    """

    kwargs = _get_kwargs(
        group_name=group_name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    group_name: None | str | Unset = UNSET,
) -> HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item] | None:
    """List Config Schemas

    Args:
        group_name (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[ConfigSchemasListConfigSchemasResponse200Item]
    """

    return (
        await asyncio_detailed(
            client=client,
            group_name=group_name,
        )
    ).parsed
