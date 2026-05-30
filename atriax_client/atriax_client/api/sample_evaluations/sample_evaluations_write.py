from http import HTTPStatus
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sample_evaluation import SampleEvaluation
from ...models.sample_evaluation_create import SampleEvaluationCreate
from ...types import Response


def _get_kwargs(
    evaluation_experiment_id: UUID,
    *,
    body: list[SampleEvaluationCreate] | SampleEvaluationCreate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/evaluation_experiments/{evaluation_experiment_id}/sample_evaluations/".format(
            evaluation_experiment_id=quote(str(evaluation_experiment_id), safe=""),
        ),
    }

    if isinstance(body, list):
        _kwargs["json"] = []
        for body_type_0_item_data in body:
            body_type_0_item = body_type_0_item_data.to_dict()
            _kwargs["json"].append(body_type_0_item)

    else:
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[SampleEvaluation] | SampleEvaluation | None:
    if response.status_code == 200:

        def _parse_response_200(data: object) -> list[SampleEvaluation] | SampleEvaluation:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                response_200_type_0 = []
                _response_200_type_0 = data
                for response_200_type_0_item_data in _response_200_type_0:
                    response_200_type_0_item = SampleEvaluation.from_dict(response_200_type_0_item_data)

                    response_200_type_0.append(response_200_type_0_item)

                return response_200_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_200_type_1 = SampleEvaluation.from_dict(data)

            return response_200_type_1

        response_200 = _parse_response_200(response.json())

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
) -> Response[HTTPValidationError | list[SampleEvaluation] | SampleEvaluation]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[SampleEvaluationCreate] | SampleEvaluationCreate,
) -> Response[HTTPValidationError | list[SampleEvaluation] | SampleEvaluation]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (list[SampleEvaluationCreate] | SampleEvaluationCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[SampleEvaluation] | SampleEvaluation]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[SampleEvaluationCreate] | SampleEvaluationCreate,
) -> HTTPValidationError | list[SampleEvaluation] | SampleEvaluation | None:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (list[SampleEvaluationCreate] | SampleEvaluationCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[SampleEvaluation] | SampleEvaluation
    """

    return sync_detailed(
        evaluation_experiment_id=evaluation_experiment_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[SampleEvaluationCreate] | SampleEvaluationCreate,
) -> Response[HTTPValidationError | list[SampleEvaluation] | SampleEvaluation]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (list[SampleEvaluationCreate] | SampleEvaluationCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[SampleEvaluation] | SampleEvaluation]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[SampleEvaluationCreate] | SampleEvaluationCreate,
) -> HTTPValidationError | list[SampleEvaluation] | SampleEvaluation | None:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (list[SampleEvaluationCreate] | SampleEvaluationCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[SampleEvaluation] | SampleEvaluation
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            client=client,
            body=body,
        )
    ).parsed
