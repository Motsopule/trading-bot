"""
Trade journal module: append-only CSV persistence for completed trades.

Provides professional-grade trade analytics persistence without altering
trading, strategy, risk, or execution logic.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Optional

JOURNAL_FILENAME = "trade_journal.csv"
CSV_HEADER = [
    "timestamp",
    "symbol",
    "timeframe",
    "execution_mode",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "pnl_percent",
    "exit_reason",
    "trade_duration",
]


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse ISO format timestamp; return None on failure."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _trade_duration_seconds(entry_time_str: str, exit_time_str: str) -> Optional[float]:
    """
    Compute trade duration in seconds from ISO entry/exit times.

    Uses existing trade_result fields only; no duplicate calculations.
    Returns None if parsing fails.
    """
    entry_dt = _parse_iso(entry_time_str) if entry_time_str else None
    exit_dt = _parse_iso(exit_time_str) if exit_time_str else None
    if entry_dt is None or exit_dt is None:
        return None
    return (exit_dt - entry_dt).total_seconds()


def write_trade_entry(
    trade_result: Dict,
    symbol: str,
    timeframe: str,
    execution_mode: str,
    logger=None,
) -> bool:
    """
    Append one completed trade to the persistent CSV journal.

    Uses only the existing trade_result dictionary; no duplicate PnL or
    price calculations. Append-only; creates file with header if missing.
    Safe on file lock or missing path: logs and returns False without raising.

    Args:
        trade_result: Dict with entry_price, exit_price, quantity, pnl,
            pnl_percent, exit_reason, entry_time, exit_time (ISO strings).
        symbol: Trading pair (e.g. ETHUSDT).
        timeframe: Candle timeframe (e.g. 4h).
        execution_mode: 'PAPER' or 'LIVE'.
        logger: Optional logger for JOURNAL_WRITE and errors.

    Returns:
        True if the row was written, False on any write/lock error.
    """
    try:
        timestamp = trade_result.get("exit_time") or datetime.now(timezone.utc).isoformat()
        entry_price = trade_result.get("entry_price", "")
        exit_price = trade_result.get("exit_price", "")
        quantity = trade_result.get("quantity", "")
        pnl = trade_result.get("pnl", "")
        pnl_percent = trade_result.get("pnl_percent", "")
        exit_reason = trade_result.get("exit_reason", "")

        duration = _trade_duration_seconds(
            trade_result.get("entry_time", ""),
            trade_result.get("exit_time", ""),
        )
        trade_duration = "" if duration is None else str(round(duration, 2))

        row = {
            "timestamp": timestamp,
            "symbol": symbol,
            "timeframe": timeframe,
            "execution_mode": execution_mode,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "exit_reason": exit_reason,
            "trade_duration": trade_duration,
        }

        file_exists = os.path.isfile(JOURNAL_FILENAME)
        with open(JOURNAL_FILENAME, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        if logger:
            logger.info(
                "symbol=%s timeframe=%s state=JOURNAL_WRITE file=%s",
                symbol, timeframe, JOURNAL_FILENAME,
            )
        return True

    except (OSError, IOError) as e:
        if logger:
            logger.warning(
                "symbol=%s timeframe=%s state=JOURNAL_WRITE error=file_locked_or_missing msg=%s",
                symbol, timeframe, str(e),
            )
        return False
