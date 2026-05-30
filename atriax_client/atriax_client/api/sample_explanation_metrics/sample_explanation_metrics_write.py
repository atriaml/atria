from http import HTTPStatus
from typing import Any, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sample_explanation_metric import SampleExplanationMetric
from ...models.sample_explanation_metric_create import SampleExplanationMetricCreate
from ...types import UNSET, Response


def _get_kwargs(
    evaluation_experiment_id: UUID,
    *,
    body: Union["SampleExplanationMetricCreate", list["SampleExplanationMetricCreate"]],
    sample_explanation_id: UUID,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    json_sample_explanation_id = str(sample_explanation_id)
    params["sample_explanation_id"] = json_sample_explanation_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/api/v1/evaluations/{evaluation_experiment_id}/sample_explanation_metrics/",
        "params": params,
    }

    _kwargs["json"]: dict[str, Any] | list[dict[str, Any]]
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
) -> HTTPValidationError | Union["SampleExplanationMetric", list["SampleExplanationMetric"]] | None:
    if response.status_code == 200:

        def _parse_response_200(data: object) -> Union["SampleExplanationMetric", list["SampleExplanationMetric"]]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                response_200_type_0 = []
                _response_200_type_0 = data
                for response_200_type_0_item_data in _response_200_type_0:
                    response_200_type_0_item = SampleExplanationMetric.from_dict(response_200_type_0_item_data)

                    response_200_type_0.append(response_200_type_0_item)

                return response_200_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_200_type_1 = SampleExplanationMetric.from_dict(data)

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
) -> Response[HTTPValidationError | Union["SampleExplanationMetric", list["SampleExplanationMetric"]]]:
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
    body: Union["SampleExplanationMetricCreate", list["SampleExplanationMetricCreate"]],
    sample_explanation_id: UUID,
) -> Response[HTTPValidationError | Union["SampleExplanationMetric", list["SampleExplanationMetric"]]]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (UUID):
        body (Union['SampleExplanationMetricCreate', list['SampleExplanationMetricCreate']]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['SampleExplanationMetric', list['SampleExplanationMetric']]]]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        body=body,
        sample_explanation_id=sample_explanation_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: Union["SampleExplanationMetricCreate", list["SampleExplanationMetricCreate"]],
    sample_explanation_id: UUID,
) -> HTTPValidationError | Union["SampleExplanationMetric", list["SampleExplanationMetric"]] | None:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (UUID):
        body (Union['SampleExplanationMetricCreate', list['SampleExplanationMetricCreate']]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['SampleExplanationMetric', list['SampleExplanationMetric']]]
    """

    return sync_detailed(
        evaluation_experiment_id=evaluation_experiment_id,
        client=client,
        body=body,
        sample_explanation_id=sample_explanation_id,
    ).parsed


async def asyncio_detailed(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: Union["SampleExplanationMetricCreate", list["SampleExplanationMetricCreate"]],
    sample_explanation_id: UUID,
) -> Response[HTTPValidationError | Union["SampleExplanationMetric", list["SampleExplanationMetric"]]]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (UUID):
        body (Union['SampleExplanationMetricCreate', list['SampleExplanationMetricCreate']]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['SampleExplanationMetric', list['SampleExplanationMetric']]]]
    """

    kwargs = _get_kwargs(
        evaluation_experiment_id=evaluation_experiment_id,
        body=body,
        sample_explanation_id=sample_explanation_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    evaluation_experiment_id: UUID,
    *,
    client: AuthenticatedClient,
    body: Union["SampleExplanationMetricCreate", list["SampleExplanationMetricCreate"]],
    sample_explanation_id: UUID,
) -> HTTPValidationError | Union["SampleExplanationMetric", list["SampleExplanationMetric"]] | None:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        sample_explanation_id (UUID):
        body (Union['SampleExplanationMetricCreate', list['SampleExplanationMetricCreate']]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['SampleExplanationMetric', list['SampleExplanationMetric']]]
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            client=client,
            body=body,
            sample_explanation_id=sample_explanation_id,
        )
    ).parsed
