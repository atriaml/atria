from http import HTTPStatus
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.explanation_visualization_data import ExplanationVisualizationData
from ...models.http_validation_error import HTTPValidationError
from ...models.image_explanation_visualizer_options import ImageExplanationVisualizerOptions
from ...types import UNSET, Response, Unset


def _get_kwargs(
    evaluation_experiment_id: UUID,
    id: UUID,
    *,
    body: ImageExplanationVisualizerOptions | None | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/evaluations/{evaluation_experiment_id}/sample_explanations/{id}/viz_data/".format(
            evaluation_experiment_id=quote(str(evaluation_experiment_id), safe=""),
            id=quote(str(id), safe=""),
        ),
    }

    if isinstance(body, ImageExplanationVisualizerOptions):
        _kwargs["json"] = body.to_dict()
    else:
        _kwargs["json"] = body

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ExplanationVisualizationData | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = ExplanationVisualizationData.from_dict(response.json())

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
) -> Response[ExplanationVisualizationData | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    evaluation_experiment_id: UUID,
    id: UUID,
    *,
    client: AuthenticatedClient,
    body: ImageExplanationVisualizerOptions | None | Unset = UNSET,
) -> Response[ExplanationVisualizationData | HTTPValidationError]:
    """Viz Data

    Args:
        evaluation_experiment_id (UUID):
        id (UUID):
        body (ImageExplanationVisualizerOptions | None | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ExplanationVisualizationData | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        id=id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    evaluation_experiment_id: UUID,
    id: UUID,
    *,
    client: AuthenticatedClient,
    body: ImageExplanationVisualizerOptions | None | Unset = UNSET,
) -> ExplanationVisualizationData | HTTPValidationError | None:
    """Viz Data

    Args:
        evaluation_experiment_id (UUID):
        id (UUID):
        body (ImageExplanationVisualizerOptions | None | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ExplanationVisualizationData | HTTPValidationError
    """

    return sync_detailed(
        evaluation_experiment_id=evaluation_experiment_id,
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    evaluation_experiment_id: UUID,
    id: UUID,
    *,
    client: AuthenticatedClient,
    body: ImageExplanationVisualizerOptions | None | Unset = UNSET,
) -> Response[ExplanationVisualizationData | HTTPValidationError]:
    """Viz Data

    Args:
        evaluation_experiment_id (UUID):
        id (UUID):
        body (ImageExplanationVisualizerOptions | None | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ExplanationVisualizationData | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        id=id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    evaluation_experiment_id: UUID,
    id: UUID,
    *,
    client: AuthenticatedClient,
    body: ImageExplanationVisualizerOptions | None | Unset = UNSET,
) -> ExplanationVisualizationData | HTTPValidationError | None:
    """Viz Data

    Args:
        evaluation_experiment_id (UUID):
        id (UUID):
        body (ImageExplanationVisualizerOptions | None | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ExplanationVisualizationData | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            id=id,
            client=client,
            body=body,
        )
    ).parsed
