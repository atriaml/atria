from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_sample_explanations_write import BodySampleExplanationsWrite
from ...models.http_validation_error import HTTPValidationError
from ...models.sample_explanation_write_response import SampleExplanationWriteResponse
from ...types import Response


def _get_kwargs(
    evaluation_experiment_id: UUID,
    *,
    body: BodySampleExplanationsWrite,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/api/v1/evaluations/{evaluation_experiment_id}/sample_explanations/",
    }

    _kwargs["files"] = body.to_multipart()

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[
    Union[HTTPValidationError, Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]]
]:
    if response.status_code == 200:

        def _parse_response_200(
            data: object,
        ) -> Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                response_200_type_0 = []
                _response_200_type_0 = data
                for response_200_type_0_item_data in _response_200_type_0:
                    response_200_type_0_item = SampleExplanationWriteResponse.from_dict(response_200_type_0_item_data)

                    response_200_type_0.append(response_200_type_0_item)

                return response_200_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_200_type_1 = SampleExplanationWriteResponse.from_dict(data)

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
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[
    Union[HTTPValidationError, Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]]
]:
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
    body: BodySampleExplanationsWrite,
) -> Response[
    Union[HTTPValidationError, Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]]
]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (BodySampleExplanationsWrite):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['SampleExplanationWriteResponse', list['SampleExplanationWriteResponse']]]]
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
    body: BodySampleExplanationsWrite,
) -> Optional[
    Union[HTTPValidationError, Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]]
]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (BodySampleExplanationsWrite):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['SampleExplanationWriteResponse', list['SampleExplanationWriteResponse']]]
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
    body: BodySampleExplanationsWrite,
) -> Response[
    Union[HTTPValidationError, Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]]
]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (BodySampleExplanationsWrite):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['SampleExplanationWriteResponse', list['SampleExplanationWriteResponse']]]]
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
    body: BodySampleExplanationsWrite,
) -> Optional[
    Union[HTTPValidationError, Union["SampleExplanationWriteResponse", list["SampleExplanationWriteResponse"]]]
]:
    """Write

    Args:
        evaluation_experiment_id (UUID):
        body (BodySampleExplanationsWrite):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['SampleExplanationWriteResponse', list['SampleExplanationWriteResponse']]]
    """

    return (
        await asyncio_detailed(
            evaluation_experiment_id=evaluation_experiment_id,
            client=client,
            body=body,
        )
    ).parsed
