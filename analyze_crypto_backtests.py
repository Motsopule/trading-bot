"""
Analyze existing crypto backtest trade logs and produce a validation summary.

Loads all backtest_trades_*USDT.csv files, computes per-asset metrics
(Total Trades, Win Rate, Profit Factor, Average Trade Return, Max Drawdown,
Sharpe Ratio), writes crypto_validation_summary.csv and prints a comparison table.

This script is for research and validation only; it does not modify live trading.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Optional

import pandas as pd


# Default capital used to reconstruct equity for drawdown/Sharpe when not in CSV
DEFAULT_INITIAL_CAPITAL = 1000.0


def find_crypto_backtest_csvs(directory: Optional[str] = None) -> List[str]:
    """
    Find all crypto backtest trade CSV files (backtest_trades_*USDT.csv).

    Args:
        directory: Directory to search; defaults to script directory.

    Returns:
        Sorted list of full paths to matching CSV files.
    """
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(directory, "backtest_trades_*USDT.csv")
    files = sorted(glob.glob(pattern))
    return files


def load_trades(csv_path: str) -> pd.DataFrame:
    """
    Load a single backtest trade log CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with columns expected from backtest.py trade log.
    """
    df = pd.read_csv(csv_path)
    required = {"pnl", "return_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    return df


def compute_equity_curve(
    trades: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> List[float]:
    """
    Build equity curve from trade PnL sequence (cumulative).

    Args:
        trades: DataFrame with 'pnl' column in chronological order.
        initial_capital: Starting equity.

    Returns:
        List of equity values after each trade (length = len(trades)).
    """
    if trades.empty:
        return []
    cum = trades["pnl"].cumsum()
    curve = (initial_capital + cum).tolist()
    return curve


def compute_max_drawdown_from_equity(equity_curve: List[float]) -> float:
    """
    Maximum drawdown in percentage from an equity curve.

    Args:
        equity_curve: Sequence of equity values.

    Returns:
        Max drawdown as a negative percentage (e.g. -12.5 for -12.5%).
    """
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe_from_trades(
    trades: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Estimate annualized Sharpe ratio from trade returns and time span.

    Uses trade return_pct and entry/exit times to infer period length,
    then annualizes assuming zero risk-free rate unless specified.

    Args:
        trades: DataFrame with 'return_pct', 'entry_time', 'exit_time'.
        initial_capital: Used to convert PnL to return if needed.
        risk_free_rate: Annual risk-free rate (decimal).

    Returns:
        Annualized Sharpe ratio; 0.0 if undefined (e.g. no variance or no trades).
    """
    if trades.empty or len(trades) < 2:
        return 0.0
    returns_pct = trades["return_pct"].astype(float)
    mean_ret = returns_pct.mean()
    std_ret = returns_pct.std()
    if std_ret == 0 or pd.isna(std_ret):
        return 0.0
    # Time span in years from first entry to last exit
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        try:
            first_ts = pd.to_datetime(trades["entry_time"].iloc[0])
            last_ts = pd.to_datetime(trades["exit_time"].iloc[-1])
            years = (last_ts - first_ts).total_seconds() / (365.25 * 24 * 3600)
            years = max(years, 1 / 365.25)  # avoid div by zero
        except Exception:
            years = 1.0
    else:
        years = 1.0
    trades_per_year = len(trades) / years
    # Excess return per trade (percentage); annualized: mean * trades_per_year
    # Sharpe = (mean - rf_per_trade) / std * sqrt(trades_per_year)
    rf_per_trade = risk_free_rate / trades_per_year * 100.0  # in % points
    sharpe = (mean_ret - rf_per_trade) / std_ret * math.sqrt(trades_per_year)
    return float(sharpe)


def metrics_for_asset(
    asset: str,
    trades: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> Dict[str, float]:
    """
    Compute all required metrics for one asset's trade log.

    Args:
        asset: Asset symbol (e.g. 'ETHUSDT').
        trades: DataFrame of trades with pnl, return_pct, etc.
        initial_capital: Starting capital for equity/drawdown.

    Returns:
        Dictionary with total_trades, win_rate, profit_factor, avg_trade_return_pct,
        max_drawdown, sharpe_ratio.
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "asset": asset,
            "total_trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_return_pct": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    gross_profit = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())
    win_rate = len(wins) / total_trades * 100.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    avg_trade_return_pct = float(trades["return_pct"].mean())
    equity_curve = compute_equity_curve(trades, initial_capital)
    max_drawdown = compute_max_drawdown_from_equity(equity_curve)
    sharpe_ratio = compute_sharpe_from_trades(trades, initial_capital)

    return {
        "asset": asset,
        "total_trades": float(total_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return_pct": avg_trade_return_pct,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def run_analysis(
    csv_directory: Optional[str] = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    output_csv: str = "crypto_validation_summary.csv",
) -> pd.DataFrame:
    """
    Load all crypto backtest CSVs, compute metrics, save summary CSV, print table.

    Args:
        csv_directory: Directory containing backtest_trades_*USDT.csv files.
        initial_capital: Capital assumed for drawdown/Sharpe.
        output_csv: Path for output summary CSV.

    Returns:
        DataFrame of per-asset metrics.
    """
    files = find_crypto_backtest_csvs(csv_directory)
    if not files:
        dir_used = csv_directory or os.path.dirname(os.path.abspath(__file__))
        raise FileNotFoundError(
            f"No files matching backtest_trades_*USDT.csv in {dir_used}"
        )

    results: List[Dict[str, float]] = []
    for path in files:
        basename = os.path.basename(path)
        # e.g. backtest_trades_ETHUSDT.csv -> ETHUSDT
        asset = basename.replace("backtest_trades_", "").replace(".csv", "")
        try:
            trades = load_trades(path)
        except Exception as e:
            print(f"Warning: skipped {path}: {e}")
            continue
        m = metrics_for_asset(asset, trades, initial_capital)
        results.append(m)

    if not results:
        print("No valid crypto backtest files could be loaded.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    out_path = output_csv
    if csv_directory and not os.path.isabs(output_csv):
        out_path = os.path.join(csv_directory, output_csv)
    df.to_csv(out_path, index=False)
    print(f"Summary saved to '{out_path}'.")

    print_formatted_table(results)
    return df


def print_formatted_table(results: List[Dict[str, float]]) -> None:
    """
    Print a formatted comparison table of crypto backtest metrics to the terminal.

    Args:
        results: List of per-asset metric dictionaries.
    """
    header = (
        f"{'Asset':<12}"
        f"{'Trades':>8}"
        f"{'WinRate':>10}"
        f"{'PF':>8}"
        f"{'AvgRet%':>10}"
        f"{'MaxDD%':>10}"
        f"{'Sharpe':>8}"
    )
    print("\n--- Crypto backtest validation summary ---\n")
    print(header)
    print("-" * len(header))

    for row in results:
        pf_val = row["profit_factor"]
        pf_str = "Inf" if math.isinf(pf_val) else f"{pf_val:.2f}"
        line = (
            f"{row['asset']:<12}"
            f"{int(row['total_trades']):>8}"
            f"{row['win_rate']:>9.1f}%"
            f"{pf_str:>8}"
            f"{row['avg_trade_return_pct']:>9.2f}%"
            f"{row['max_drawdown']:>9.1f}%"
            f"{row['sharpe_ratio']:>8.2f}"
        )
        print(line)
    print()


if __name__ == "__main__":
    run_analysis(output_csv="crypto_validation_summary.csv")
