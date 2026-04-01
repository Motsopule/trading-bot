"""
Cross-asset validation: run the same strategy across all assets and collect metrics.

Runs backtests for crypto (Binance), forex, indices, and commodities (yfinance),
then writes Profit Factor, Win Rate, Max Drawdown, Sharpe Ratio, and Total Trades
to cross_asset_validation_results.csv.

This script is for research and validation only.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from multi_backtest import run_multi_backtest


def run_cross_asset_validation(
    lookback_crypto: int = 8,
    lookback_traditional: int = 10,
    output_csv: str = "cross_asset_validation_results.csv",
    crypto_only: bool = False,
) -> pd.DataFrame:
    """
    Run the strategy across all configured assets and save performance metrics.

    Args:
        lookback_crypto: Years of history for crypto (max available).
        lookback_traditional: Years of history for forex/indices/commodities (e.g. 10).
        output_csv: Path for output CSV.
        crypto_only: If True, run only crypto pairs.

    Returns:
        DataFrame with columns: asset, total_trades, win_rate, profit_factor,
        max_drawdown, sharpe_ratio, total_return.
    """
    results = run_multi_backtest(
        lookback_crypto=lookback_crypto,
        lookback_traditional=lookback_traditional,
        crypto_only=crypto_only,
        include_forex=not crypto_only,
        include_indices=not crypto_only,
        include_commodities=not crypto_only,
    )

    if not results:
        print("No backtest results to write.")
        return pd.DataFrame()

    # Normalize keys for CSV (pair -> asset, ensure sharpe_ratio present)
    rows: List[Dict[str, float]] = []
    for r in results:
        row = {
            "asset": r["pair"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "profit_factor": r["profit_factor"],
            "max_drawdown": r["max_drawdown"],
            "sharpe_ratio": r.get("sharpe_ratio", 0.0),
            "total_return": r["total_return"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = output_csv
    if not os.path.isabs(out_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(base_dir, output_csv)
    df.to_csv(out_path, index=False)
    print(f"Cross-asset validation results saved to '{out_path}'.")
    return df


if __name__ == "__main__":
    run_cross_asset_validation(
        lookback_crypto=8,
        lookback_traditional=10,
        output_csv="cross_asset_validation_results.csv",
        crypto_only=False,
    )
