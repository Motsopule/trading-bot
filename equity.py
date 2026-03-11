"""
Equity curve logging module.

Appends equity snapshots to logs/equity_curve.csv for tracking capital,
daily PnL, open positions count, and exposure over time.
"""

import csv
from pathlib import Path
from datetime import datetime, timezone

HEADER = ("timestamp", "capital", "daily_pnl", "positions", "exposure")
LOG_DIR = Path(__file__).resolve().parent / "logs"
EQUITY_FILE = LOG_DIR / "equity_curve.csv"


def log_equity(
    capital: float,
    daily_pnl: float,
    open_positions: int,
    exposure: float,
) -> None:
    """
    Append one row to logs/equity_curve.csv.

    Creates the file and writes the header if the file does not exist.
    Uses UTC timestamps. Append-only; never overwrites existing data.

    Args:
        capital: Current capital (e.g. from risk manager).
        daily_pnl: Daily P&L to date.
        open_positions: Number of open positions.
        exposure: Total exposure in USDT.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = EQUITY_FILE.exists()

    with open(EQUITY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADER)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        writer.writerow((ts, capital, daily_pnl, open_positions, exposure))
