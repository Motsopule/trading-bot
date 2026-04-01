"""
Retry policy for Binance, Massive (requests), and yfinance.download.

Max 3 attempts; backoff 2s, 5s, 10s on connection, timeout, and DNS failures.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

MAX_RETRIES = 3
_BACKOFF_SECS = (2, 5, 10)

try:
    from urllib3.exceptions import NameResolutionError
except ImportError:
    NameResolutionError = None  # type: ignore[misc, assignment]

_HTTP_GET_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)
if NameResolutionError is not None:
    _HTTP_GET_RETRY_EXCEPTIONS = _HTTP_GET_RETRY_EXCEPTIONS + (NameResolutionError,)


def http_get_with_retries(url: str, **kwargs: Any) -> requests.Response:
    """GET with retries on connection errors, timeouts, and DNS resolution failures."""
    req_kwargs = dict(kwargs)
    req_kwargs.setdefault("timeout", 30)
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, **req_kwargs)
            response.raise_for_status()
            return response
        except _HTTP_GET_RETRY_EXCEPTIONS:
            if attempt == 2:
                raise
            time.sleep(_BACKOFF_SECS[attempt])
    raise RuntimeError("http_get_with_retries: unreachable")


def yfinance_download_with_retries(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """yfinance.download with the same retry policy."""
    import yfinance as yf

    for attempt in range(MAX_RETRIES):
        try:
            return yf.download(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == 2:
                raise
            time.sleep(_BACKOFF_SECS[attempt])
    raise RuntimeError("yfinance_download_with_retries: unreachable")
