import logging

import httpx
from grpc import RpcError, StatusCode
from grpc.aio import AioRpcError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
MAX_WAIT_S = 60
RETRY_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_GRPC_CODES = frozenset(
    {
        StatusCode.UNAVAILABLE,
        StatusCode.DEADLINE_EXCEEDED,
        StatusCode.RESOURCE_EXHAUSTED,
        StatusCode.INTERNAL,
        StatusCode.ABORTED,
    }
)


def is_transient(error: BaseException) -> bool:
    """True for network failures and server overload, false for real request errors."""
    match error:
        case UnexpectedResponse():
            return error.status_code in RETRY_HTTP_STATUS
        case httpx.HTTPStatusError():
            return error.response.status_code in RETRY_HTTP_STATUS
        case AioRpcError():
            return error.code() in RETRY_GRPC_CODES
        case RpcError():
            return error.code() in RETRY_GRPC_CODES  # ty: ignore[unresolved-attribute]
        case ResponseHandlingException() | httpx.TransportError():
            return True
        case TimeoutError() | ConnectionError():
            return True
        case _:
            return False


def is_already_exists(error: BaseException) -> bool:
    """True when the server says the resource exists (HTTP 409 or gRPC ALREADY_EXISTS)."""
    match error:
        case UnexpectedResponse():
            return error.status_code == 409
        case AioRpcError():
            return error.code() == StatusCode.ALREADY_EXISTS
        case _:
            return False


with_retry = retry(
    retry=retry_if_exception(is_transient),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_random_exponential(multiplier=1, max=MAX_WAIT_S),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
