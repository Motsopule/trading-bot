"""
Forex OHLCV from Massive.com aggregates API (research-only).

Uses GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
with MASSIVE_API_KEY (Bearer). Pagination follows next_url when present.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import pandas as pd

from research.constants import LOOKBACK_TRADITIONAL_YEARS
from research.data_providers.http_retry import http_get_with_retries
from research.data_providers.utc_range import utc_end_and_start

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("MASSIVE_API_BASE", "https://api.massive.com").rstrip("/")

# Pause after each successful page fetch to reduce API rate pressure.
REQUEST_DELAY = 0.5

# Forex pairs that use Massive instead of yfinance in research backtests.
FOREX_MASSIVE_SYMBOLS = frozenset({"EURUSD", "GBPUSD", "USDJPY"})


def _massive_ticker(symbol: str) -> str:
    """Map EURUSD -> C:EURUSD."""
    s = symbol.upper().strip()
    if s.startswith("C:"):
        return s
    return f"C:{s}"


def _timeframe_to_massive(timeframe: str) -> tuple[int, str]:
    """Map research timeframe strings to Massive multiplier + timespan."""
    mapping: dict[str, tuple[int, str]] = {
        "15m": (15, "minute"),
        "1h": (1, "hour"),
        "4h": (4, "hour"),
        "1d": (1, "day"),
    }
    if timeframe not in mapping:
        raise ValueError(
            f"Unsupported timeframe for Massive forex: {timeframe!r} "
            f"(expected one of {list(mapping)})"
        )
    return mapping[timeframe]


def _massive_headers() -> Dict[str, str]:
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise ValueError(
            "MASSIVE_API_KEY is not set. Add it to your environment or .env file."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def fetch_massive_forex_ohlcv(
    symbol: str,
    timeframe: str,
    lookback_years: int = LOOKBACK_TRADITIONAL_YEARS,
) -> pd.DataFrame:
    """
    Fetch forex OHLCV from Massive aggregates API.

    Window is always computed internally from current UTC (no external dates).

    Args:
        symbol: Pair without prefix, e.g. \"EURUSD\", \"GBPUSD\".
        timeframe: One of \"15m\", \"1h\", \"4h\", \"1d\".
        lookback_years: Years of history (end = now UTC, start = end - 365 * lookback_years days).

    Returns:
        DataFrame with columns open, high, low, close, volume and a UTC datetime index.
    """
    end_date, start_date = utc_end_and_start(lookback_years)
    if end_date > datetime.utcnow():
        raise ValueError(f"Invalid end_date detected: {end_date}")
    start_date_str = start_date.date().isoformat()
    end_date_str = end_date.date().isoformat()
    logger.info(
        f"[FINAL RANGE] {symbol} {timeframe} → {start_date_str} to {end_date_str}"
    )

    ticker = _massive_ticker(symbol)
    multiplier, timespan = _timeframe_to_massive(timeframe)
    ticker_path = quote(ticker, safe="")
    path = (
        f"/v2/aggs/ticker/{ticker_path}/range/"
        f"{multiplier}/{timespan}/{start_date_str}/{end_date_str}"
    )
    headers = _massive_headers()

    all_results: List[Dict[str, Any]] = []
    current_url: Optional[str] = f"{BASE_URL}{path}"
    params: Optional[Dict[str, Any]] = {"limit": 50000, "sort": "asc"}
    first = True

    while current_url:
        resp = http_get_with_retries(
            current_url,
            headers=headers,
            params=params if first else None,
            timeout=120,
        )
        first = False
        params = None
        data = resp.json()

        status = data.get("status")
        if status and status not in ("OK", "DELAYED"):
            logger.warning("Massive API status: %s", status)

        batch = data.get("results") or []
        all_results.extend(batch)

        time.sleep(REQUEST_DELAY)

        next_url = data.get("next_url")
        if not next_url:
            break
        if next_url.startswith("http"):
            current_url = next_url
        else:
            current_url = urljoin(BASE_URL + "/", next_url.lstrip("/"))

    if not all_results:
        raise ValueError(
            f"No Massive aggregate data returned for {ticker} "
            f"{timeframe} from {start_date_str} to {end_date_str}"
        )

    df = pd.DataFrame(all_results)
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Massive response missing column {col!r}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)

    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df
