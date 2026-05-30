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
    config_type: ConfigType | None | Unset = UNSET,
    name: None | Unset | str = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_config_type: None | Unset | str
    if isinstance(config_type, Unset):
        json_config_type = UNSET
    elif isinstance(config_type, ConfigType):
        json_config_type = config_type.value
    else:
        json_config_type = config_type
    params["config_type"] = json_config_type

    json_name: None | Unset | str
    if isinstance(name, Unset):
        json_name = UNSET
    else:
        json_name = name
    params["name"] = json_name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/configs/find_one/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Config | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Config.from_dict(response.json())

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
) -> Response[Config | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | Unset | str = UNSET,
) -> Response[Config | HTTPValidationError]:
    """Find One

    Args:
        config_type (Union[ConfigType, None, Unset]):
        name (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Config, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
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
    config_type: ConfigType | None | Unset = UNSET,
    name: None | Unset | str = UNSET,
) -> Config | HTTPValidationError | None:
    """Find One

    Args:
        config_type (Union[ConfigType, None, Unset]):
        name (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Config, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        config_type=config_type,
        name=name,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | Unset | str = UNSET,
) -> Response[Config | HTTPValidationError]:
    """Find One

    Args:
        config_type (Union[ConfigType, None, Unset]):
        name (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Config, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        config_type=config_type,
        name=name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    config_type: ConfigType | None | Unset = UNSET,
    name: None | Unset | str = UNSET,
) -> Config | HTTPValidationError | None:
    """Find One

    Args:
        config_type (Union[ConfigType, None, Unset]):
        name (Union[None, Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Config, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            config_type=config_type,
            name=name,
        )
    ).parsed
