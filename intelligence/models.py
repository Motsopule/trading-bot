"""Helpers for Phase 5 context (e.g. correlation inputs)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_returns_cache: Dict[Tuple[str, ...], Optional[np.ndarray]] = {}
_returns_cache_ts: Dict[Tuple[str, ...], float] = {}

CACHE_TTL_SECONDS = 300  # 5 minutes


def clear_returns_cache() -> None:
    """Clear TTL cache (e.g. tests)."""
    _returns_cache.clear()
    _returns_cache_ts.clear()


def _compute_returns_matrix(
    symbols: List[str],
    price_data: Dict[str, Any],
    lookback: int = 60,
) -> Optional[np.ndarray]:
    """
    Stack per-symbol simple returns into shape (n_symbols, n_periods) for np.corrcoef
    (each row is one symbol).
    """
    if not symbols:
        return None
    rows: List[np.ndarray] = []
    for sym in symbols:
        df = price_data.get(sym)
        if df is None or len(df) < lookback + 1:
            return None
        closes = df["close"].astype(float).values
        tail = closes[-(lookback + 1) :]
        prev = tail[:-1]
        rets = (tail[1:] - prev) / np.where(prev != 0, prev, np.nan)
        if not np.all(np.isfinite(rets)):
            return None
        rows.append(rets.astype(float))
    n = min(len(r) for r in rows)
    if n < 2:
        return None
    aligned = [r[-n:] for r in rows]
    return np.vstack(aligned)


def build_returns_matrix(
    symbols: List[str],
    price_data: Dict[str, Any],
    lookback: int = 60,
) -> Optional[np.ndarray]:
    """
    Cached wrapper around _compute_returns_matrix.
    Cache key is the symbol set; entries expire after CACHE_TTL_SECONDS.
    """
    now = time.time()
    cache_key = tuple(symbols)

    if cache_key in _returns_cache:
        if now - _returns_cache_ts[cache_key] < CACHE_TTL_SECONDS:
            return _returns_cache[cache_key]

    matrix = _compute_returns_matrix(symbols, price_data, lookback=lookback)
    _returns_cache[cache_key] = matrix
    _returns_cache_ts[cache_key] = now
    return matrix
