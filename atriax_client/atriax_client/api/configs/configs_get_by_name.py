from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.config import Config
from ...models.config_type import ConfigType
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    config_type: ConfigType,
    name: str,
    variant: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/configs/{config_type}/{name}/{variant}/",
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
    config_type: ConfigType,
    name: str,
    variant: str,
    *,
    client: AuthenticatedClient,
) -> Response[Config | HTTPValidationError]:
    """Get By Name

    Args:
        config_type (ConfigType):
        name (str):
        variant (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Config, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        config_type=config_type,
        name=name,
        variant=variant,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    config_type: ConfigType,
    name: str,
    variant: str,
    *,
    client: AuthenticatedClient,
) -> Config | HTTPValidationError | None:
    """Get By Name

    Args:
        config_type (ConfigType):
        name (str):
        variant (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Config, HTTPValidationError]
    """

    return sync_detailed(
        config_type=config_type,
        name=name,
        variant=variant,
        client=client,
    ).parsed


async def asyncio_detailed(
    config_type: ConfigType,
    name: str,
    variant: str,
    *,
    client: AuthenticatedClient,
) -> Response[Config | HTTPValidationError]:
    """Get By Name

    Args:
        config_type (ConfigType):
        name (str):
        variant (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Config, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        config_type=config_type,
        name=name,
        variant=variant,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    config_type: ConfigType,
    name: str,
    variant: str,
    *,
    client: AuthenticatedClient,
) -> Config | HTTPValidationError | None:
    """Get By Name

    Args:
        config_type (ConfigType):
        name (str):
        variant (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Config, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            config_type=config_type,
            name=name,
            variant=variant,
            client=client,
        )
    ).parsed
