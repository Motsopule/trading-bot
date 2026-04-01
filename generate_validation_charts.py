"""
Generate validation charts from backtest results: equity curve per asset,
drawdown curve, and portfolio equity (all assets traded simultaneously).

Saves charts to research_results/. For research and validation only.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from multi_backtest import run_multi_backtest_with_equity


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown (percentage below running peak) from an equity series.

    Args:
        equity: Equity values with datetime index.

    Returns:
        Series of drawdown in percentage (0 to -inf).
    """
    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100.0
    return dd


def _align_equity_curves(equity_curves: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Align multiple equity series to a common datetime index (union), forward-fill.

    Args:
        equity_curves: Dict of asset -> equity Series.

    Returns:
        DataFrame with shared index, one column per asset, forward-filled.
    """
    if not equity_curves:
        return pd.DataFrame()
    all_index = equity_curves[next(iter(equity_curves))].index
    for eq in equity_curves.values():
        all_index = all_index.union(eq.index)
    all_index = all_index.sort_values()
    df = pd.DataFrame(index=all_index)
    for name, eq in equity_curves.items():
        df[name] = eq.reindex(all_index, method="ffill")
    return df


def _portfolio_equity_equal_weight(equity_df: pd.DataFrame) -> pd.Series:
    """
    Portfolio equity as equal-weight combination: each asset normalized to
    start at 1.0, then mean across assets at each time, scaled by 1000.

    Args:
        equity_df: DataFrame of aligned equity curves (columns = assets).

    Returns:
        Series of portfolio equity.
    """
    if equity_df.empty:
        return pd.Series(dtype=float)
    first = equity_df.iloc[0].replace(0, float("nan"))
    norm = equity_df / first
    norm = norm.ffill().fillna(1.0)
    portfolio = norm.mean(axis=1) * 1000.0
    return portfolio


def generate_charts(
    lookback_crypto: int = 8,
    lookback_traditional: int = 10,
    output_dir: str = "research_results",
    crypto_only: bool = False,
) -> None:
    """
    Run backtests, then generate and save equity, drawdown, and portfolio charts.

    Args:
        lookback_crypto: Years of history for crypto.
        lookback_traditional: Years of history for traditional assets.
        output_dir: Directory to save PNG files.
        crypto_only: If True, run only crypto (faster).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Running backtests to collect equity curves...")
    results, equity_curves = run_multi_backtest_with_equity(
        lookback_crypto=lookback_crypto,
        lookback_traditional=lookback_traditional,
        crypto_only=crypto_only,
        include_forex=not crypto_only,
        include_indices=not crypto_only,
        include_commodities=not crypto_only,
        verbose=True,
    )

    if not equity_curves:
        print("No equity curves available; skipping charts.")
        return

    # 1) Equity curve per asset (all on one figure, normalized to 100)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for name, eq in equity_curves.items():
        if eq.empty or eq.iloc[0] == 0:
            continue
        norm = eq / eq.iloc[0] * 100.0
        ax1.plot(norm.index, norm.values, label=name, alpha=0.8)
    ax1.set_title("Equity curve per asset (normalized to 100)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity (index)")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(
        os.path.join(output_dir, "equity_curves_per_asset.png"),
        dpi=150,
    )
    plt.close(fig1)
    print(f"Saved {output_dir}/equity_curves_per_asset.png")

    # 2) Drawdown curve per asset (all on one figure)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for name, eq in equity_curves.items():
        if eq.empty:
            continue
        dd = _drawdown_series(eq)
        ax2.plot(dd.index, dd.values, label=name, alpha=0.8)
    ax2.set_title("Drawdown curve per asset (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(output_dir, "drawdown_curves.png"),
        dpi=150,
    )
    plt.close(fig2)
    print(f"Saved {output_dir}/drawdown_curves.png")

    # 3) Portfolio equity (equal weight, all assets traded simultaneously)
    aligned = _align_equity_curves(equity_curves)
    if not aligned.empty:
        portfolio_eq = _portfolio_equity_equal_weight(aligned)
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(portfolio_eq.index, portfolio_eq.values, color="navy", linewidth=2)
        ax3.set_title("Portfolio equity (equal weight, all assets)")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Portfolio equity")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(
            os.path.join(output_dir, "portfolio_equity.png"),
            dpi=150,
        )
        plt.close(fig3)
        print(f"Saved {output_dir}/portfolio_equity.png")

    print("Charts saved to", output_dir)


if __name__ == "__main__":
    generate_charts(
        lookback_crypto=8,
        lookback_traditional=10,
        output_dir="research_results",
        crypto_only=False,
    )
