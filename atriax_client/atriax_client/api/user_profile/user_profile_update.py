from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_user_profile_update import BodyUserProfileUpdate
from ...models.http_validation_error import HTTPValidationError
from ...models.user_profile import UserProfile
from ...types import UNSET, Response


def _get_kwargs(
    *,
    body: BodyUserProfileUpdate,
    user_id: UUID,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    json_user_id = str(user_id)
    params["user_id"] = json_user_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/api/v1/user_profile/update",
        "params": params,
    }

    _kwargs["files"] = body.to_multipart()

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, UserProfile]]:
    if response.status_code == 200:
        response_200 = UserProfile.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, UserProfile]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: BodyUserProfileUpdate,
    user_id: UUID,
) -> Response[Union[HTTPValidationError, UserProfile]]:
    """Update

    Args:
        user_id (UUID):
        body (BodyUserProfileUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserProfile]]
    """

    kwargs = _get_kwargs(
        body=body,
        user_id=user_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: BodyUserProfileUpdate,
    user_id: UUID,
) -> Optional[Union[HTTPValidationError, UserProfile]]:
    """Update

    Args:
        user_id (UUID):
        body (BodyUserProfileUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserProfile]
    """

    return sync_detailed(
        client=client,
        body=body,
        user_id=user_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: BodyUserProfileUpdate,
    user_id: UUID,
) -> Response[Union[HTTPValidationError, UserProfile]]:
    """Update

    Args:
        user_id (UUID):
        body (BodyUserProfileUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserProfile]]
    """

    kwargs = _get_kwargs(
        body=body,
        user_id=user_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: BodyUserProfileUpdate,
    user_id: UUID,
) -> Optional[Union[HTTPValidationError, UserProfile]]:
    """Update

    Args:
        user_id (UUID):
        body (BodyUserProfileUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserProfile]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            user_id=user_id,
        )
    ).parsed
