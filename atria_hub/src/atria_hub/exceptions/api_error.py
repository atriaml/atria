from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from atria_core.logger.logger import get_logger
from atriax_client.types import Response

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ApiResponseError(RuntimeError):
    """Generic exception for API response errors."""

    def __init__(self, request_name: str, status_code: int, content: str):
        super().__init__(
            f"API request '{request_name}' failed with status {status_code}: {content}"
        )
        self.request_name = request_name
        self.status_code = status_code
        self.content = content


def api_error_handler(func: F) -> F:
    """Decorator to handle API response errors uniformly."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function name and class name for error reporting
        class_name = (
            args[0].__class__.__name__
            if args and hasattr(args[0], "__class__")
            else "Unknown"
        )
        request_name = f"{class_name}::{func.__name__}"

        try:
            result: Response = func(*args, **kwargs)
            if result.status_code != 200:
                raise ApiResponseError(
                    request_name=request_name,
                    status_code=result.status_code,
                    content=result.content.decode("utf-8"),
                )
            parsed = result.parsed
            return parsed if parsed is not None else result.content

        except ApiResponseError:
            raise
        except Exception as e:
            # Wrap other exceptions in ApiResponseError
            logger.error(f"Unexpected error in {request_name}: {e}")
            raise ApiResponseError(
                request_name=request_name, status_code=500, content=str(e)
            ) from e

    return wrapper
