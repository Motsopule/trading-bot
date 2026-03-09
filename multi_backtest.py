"""
Run the MA crossover backtest across multiple trading pairs and summarize results.

This script reuses the existing backtest engine in `backtest.py` (including
`TradingStrategy` and risk-style position sizing) and only uses Binance's
public klines endpoint. No live trading or authenticated API is used.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from backtest import BacktestConfig, run_backtest


PAIRS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "AVAXUSDT",
]


def _load_base_config_from_env() -> BacktestConfig:
    """
    Load a base `BacktestConfig` using the same environment-driven defaults
    as `run_backtest` when called without an explicit config.

    The symbol in this base config is a placeholder and will be overridden
    per pair in the multi-asset run.

    Returns:
        BacktestConfig populated from `.env` (or built-in defaults).
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path)

    base_config = BacktestConfig(
        symbol=os.getenv("TRADING_PAIR", "ETHUSDT"),
        interval=os.getenv("TIMEFRAME", "4h"),
        initial_capital=float(os.getenv("INITIAL_CAPITAL", "1000.0")),
        atr_multiplier=float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.0")),
        daily_loss_limit_percent=float(
            os.getenv("DAILY_LOSS_LIMIT_PERCENT", "3.0")
        ),
        trading_fee_percent=float(os.getenv("TRADING_FEE_PERCENT", "0.1")),
        slippage_percent=float(os.getenv("SLIPPAGE_PERCENT", "0.05")),
    )

    return base_config


def _print_results_table(results: List[Dict[str, float]]) -> None:
    """
    Print a formatted summary table to the console.

    Args:
        results: List of per-pair result dictionaries.
    """
    header = (
        f"{'Pair':<10}"
        f"{'Trades':>8}"
        f"{'WinRate':>10}"
        f"{'PF':>8}"
        f"{'Drawdown':>12}"
        f"{'Return':>9}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        pf_val = float(row["profit_factor"])
        if math.isinf(pf_val):
            pf_str = "Inf"
        else:
            pf_str = f"{pf_val:.2f}"

        win_rate_str = f"{row['win_rate']:.0f}%"
        drawdown_str = f"{row['max_drawdown']:.0f}%"
        return_str = f"{row['total_return']:.0f}%"

        line = (
            f"{row['pair']:<10}"
            f"{int(row['total_trades']):>8}"
            f"{win_rate_str:>10}"
            f"{pf_str:>8}"
            f"{drawdown_str:>12}"
            f"{return_str:>9}"
        )
        print(line)


def _save_results_csv(results: List[Dict[str, float]], csv_path: str) -> None:
    """
    Save the per-pair backtest metrics to a CSV file.

    Args:
        results: List of per-pair result dictionaries.
        csv_path: Output CSV file path.
    """
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)


def run_multi_backtest() -> List[Dict[str, float]]:
    """
    Run the backtest engine for all configured trading pairs.

    Uses the same strategy and risk-style logic as `backtest.run_backtest`,
    fetching historical data from the same Binance klines REST API and
    applying identical parameters (timeframe, fees, slippage, capital, etc.)
    across all pairs.

    Returns:
        List of result dictionaries, one per successfully backtested pair.
    """
    base_config = _load_base_config_from_env()

    results: List[Dict[str, float]] = []
    for pair in PAIRS:
        print(f"\n=== Running backtest for {pair} ===")

        # Clone the base config but override symbol and trade log path.
        config = BacktestConfig(
            symbol=pair,
            interval=base_config.interval,
            initial_capital=base_config.initial_capital,
            atr_multiplier=base_config.atr_multiplier,
            daily_loss_limit_percent=base_config.daily_loss_limit_percent,
            risk_per_trade_percent=base_config.risk_per_trade_percent,
            max_position_exposure_percent=base_config.max_position_exposure_percent,
            lookback_years=base_config.lookback_years,
            trade_log_csv=f"backtest_trades_{pair}.csv",
            trading_fee_percent=base_config.trading_fee_percent,
            slippage_percent=base_config.slippage_percent,
        )

        try:
            _, metrics = run_backtest(config)
        except Exception as exc:  # noqa: BLE001
            print(f"Backtest failed for {pair}: {exc}")
            continue

        results.append(
            {
                "pair": pair,
                "total_trades": float(metrics["total_trades"]),
                "win_rate": float(metrics["win_rate"]),
                "profit_factor": float(metrics["profit_factor"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "total_return": float(metrics["total_return"]),
            }
        )

    if not results:
        print("No successful backtests were completed.")
        return results

    print("\nMulti-asset backtest summary:\n")
    _print_results_table(results)

    csv_path = "multi_asset_results.csv"
    _save_results_csv(results, csv_path)
    print(f"\nSummary results saved to '{csv_path}'.")

    return results


if __name__ == "__main__":
    run_multi_backtest()

