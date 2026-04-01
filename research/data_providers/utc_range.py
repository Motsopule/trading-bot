"""UTC-safe OHLCV windows: end is never after current UTC time."""

from __future__ import annotations

from datetime import datetime, timedelta


def utc_end_and_start(lookback_years: int) -> tuple[datetime, datetime]:
    """
    Return (end_date, start_date) in UTC naive datetimes.

    end_date is a single utcnow() snapshot (never computed as start + lookback).
    start_date is lookback_years * 365 days before end_date.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * lookback_years)
    return end_date, start_date
