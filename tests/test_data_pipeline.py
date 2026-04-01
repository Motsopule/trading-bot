"""
Tests for DataFetcher paginated klines pipeline (no live Binance calls).
"""

import copy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data import DataFetcher, log_event, validate_time_continuity

# 4h step in ms (Binance open times)
H4_MS = 4 * 60 * 60 * 1000


def _make_kline_row(open_ms: int, close_price: float = 100.0):
    """Single Binance kline row (list format)."""
    return [
        open_ms,
        str(close_price - 1),
        str(close_price + 1),
        str(close_price - 0.5),
        str(close_price),
        "1000.0",
        open_ms + 3599999,
        "100000.0",
        100,
        "500.0",
        "50000.0",
        "0",
    ]


def test_minimum_candles_mocked():
    """Test 1 — Minimum candles: get_klines returns at least requested count."""
    base = 1_700_000_000_000
    rows = [_make_kline_row(base + i * H4_MS) for i in range(250)]
    client = MagicMock()
    client.get_klines = MagicMock(return_value=rows[-200:])

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client
    fetcher.testnet = True
    fetcher.symbol = None
    fetcher.timeframe = None
    fetcher.consecutive_errors = 0
    fetcher.min_position_value_usdt = 10.0

    fetcher.set_trading_pair("BTCUSDT", "4h")
    df = fetcher.get_klines(limit=200)
    assert df is not None
    assert len(df) >= 200


def test_pagination_aggregates_batches():
    """Test 2 — Pagination: multiple batches aggregate to chronological series."""
    # Required 1500 > 1000 per request → two pages: 1000 newest + 500 older
    calls = []

    anchor = 1_000_000_000_000

    def side_effect(**params):
        calls.append(params.copy())
        if params.get("endTime") is None:
            return [_make_kline_row(anchor + i * H4_MS) for i in range(1000)]
        return [
            _make_kline_row(anchor - (500 - i) * H4_MS) for i in range(500)
        ]

    client = MagicMock()
    client.get_klines = MagicMock(side_effect=side_effect)

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client
    fetcher.testnet = True
    fetcher.symbol = None
    fetcher.timeframe = None
    fetcher.consecutive_errors = 0
    fetcher.min_position_value_usdt = 10.0

    klines = fetcher.get_klines_full("BTCUSDT", "4h", required=1500)
    assert len(klines) == 1500
    assert len(calls) >= 2
    assert calls[1].get("endTime") == anchor - 1
    times = [k[0] for k in klines]
    assert times == sorted(times), "Klines must be oldest→newest"


def test_insufficient_data_raises():
    """Test 3 — Failure on insufficient data: mock returns too few rows."""
    client = MagicMock()
    client.get_klines = MagicMock(return_value=[_make_kline_row(1_700_000_000_000)] * 4)

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client
    fetcher.testnet = True
    fetcher.symbol = None
    fetcher.timeframe = None
    fetcher.consecutive_errors = 0
    fetcher.min_position_value_usdt = 10.0

    with pytest.raises(Exception, match="insufficient klines"):
        fetcher.get_klines_full("BTCUSDT", "4h", required=500)


def test_close_column_no_nans():
    """Test 4 — Data consistency: close has no NaN after conversion."""
    base = 1_700_000_000_000
    rows = [_make_kline_row(base + i * H4_MS) for i in range(220)]
    client = MagicMock()
    client.get_klines = MagicMock(return_value=rows)

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client
    fetcher.testnet = True
    fetcher.symbol = None
    fetcher.timeframe = None
    fetcher.consecutive_errors = 0
    fetcher.min_position_value_usdt = 10.0

    fetcher.set_trading_pair("BTCUSDT", "4h")
    df = fetcher.get_klines(limit=200)
    assert df["close"].notna().all()


def test_deterministic_output():
    """Test 5 — Deterministic: same mock data → identical DataFrame."""
    base = 1_700_000_000_000
    rows = [_make_kline_row(base + i * H4_MS) for i in range(220)]
    client = MagicMock()
    client.get_klines = MagicMock(return_value=rows)

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client
    fetcher.testnet = True
    fetcher.symbol = None
    fetcher.timeframe = None
    fetcher.consecutive_errors = 0
    fetcher.min_position_value_usdt = 10.0

    fetcher.set_trading_pair("BTCUSDT", "4h")
    df1 = fetcher.get_klines(limit=200)
    client.get_klines = MagicMock(return_value=copy.deepcopy(rows))
    df2 = fetcher.get_klines(limit=200)

    pd.testing.assert_frame_equal(df1, df2)


@patch("data.time.sleep")
def test_safe_get_klines_retries_then_raises(mock_sleep):
    client = MagicMock()
    client.get_klines = MagicMock(side_effect=ConnectionError("network"))

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client

    with pytest.raises(ConnectionError):
        fetcher.safe_get_klines({"symbol": "BTCUSDT", "interval": "4h", "limit": 10})
    assert client.get_klines.call_count == 3
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


def test_time_continuity_gap_raises():
    """Missing candle (2x interval gap) → validate_time_continuity fails."""
    base = pd.Timestamp("2024-01-01")
    idx = [base + i * pd.Timedelta(hours=4) for i in range(50)]
    idx[25] = idx[24] + pd.Timedelta(hours=8)  # skip one 4h slot
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0},
        index=idx,
    )
    with pytest.raises(RuntimeError, match="time gap"):
        validate_time_continuity(df, "4h")


def test_duplicate_candles_raises():
    """Duplicate index timestamps → get_klines raises."""
    base = 1_700_000_000_000
    dup_rows = [_make_kline_row(base + i * H4_MS) for i in range(100)]
    dup_rows[50] = _make_kline_row(base + 49 * H4_MS)  # same open ms as row 49

    client = MagicMock()
    client.get_klines = MagicMock(return_value=dup_rows)

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client
    fetcher.testnet = True
    fetcher.symbol = None
    fetcher.timeframe = None
    fetcher.consecutive_errors = 0
    fetcher.min_position_value_usdt = 10.0

    fetcher.set_trading_pair("BTCUSDT", "4h")
    with pytest.raises(RuntimeError, match="duplicate"):
        fetcher.get_klines(limit=100)


@patch("data.time.sleep")
def test_safe_get_klines_succeeds_on_third_attempt(mock_sleep):
    client = MagicMock()
    ok = [_make_kline_row(1_700_000_000_000 + i * H4_MS) for i in range(220)]
    client.get_klines = MagicMock(
        side_effect=[ConnectionError("a"), ConnectionError("b"), ok]
    )

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client

    out = fetcher.safe_get_klines({"symbol": "BTCUSDT", "interval": "4h", "limit": 220})
    assert len(out) == 220
    assert client.get_klines.call_count == 3
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


def test_log_event_never_raises():
    class _NotJson:
        pass

    log_event({"event": "bad", "x": _NotJson()})
    log_event({"event": "ok", "n": 1})


def test_get_klines_full_requires_symbol_interval():
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = MagicMock()
    with pytest.raises(ValueError, match="symbol and interval"):
        fetcher.get_klines_full("", "4h", required=10)
    with pytest.raises(ValueError, match="symbol and interval"):
        fetcher.get_klines_full("BTCUSDT", "", required=10)


@patch("data.time.sleep")
def test_batch_none_raises(mock_sleep):
    client = MagicMock()
    client.get_klines = MagicMock(return_value=None)

    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.client = client

    with pytest.raises(RuntimeError, match="None"):
        fetcher.get_klines_full("BTCUSDT", "4h", required=10)
