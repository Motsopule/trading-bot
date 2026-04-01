"""
Crypto portfolio backtest with shared capital.

Simulates trading multiple crypto assets simultaneously using the same
MA crossover strategy and run_backtest-style logic. Uses PortfolioRiskManager
to cap total portfolio risk at 3%. Research only; does not modify production
strategy, risk, or execution modules.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from backtest import (
    BacktestConfig,
    compute_max_drawdown,
    compute_performance_metrics,
    fetch_historical_klines,
    normalize_ohlcv,
    calculate_position_size,
)
from portfolio_risk import PortfolioRiskManager
from strategy import TradingStrategy


# Crypto assets for portfolio backtest
PORTFOLIO_ASSETS = [
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "BTCUSDT",
]

# Output paths
PORTFOLIO_EQUITY_CSV = "portfolio_equity_curve.csv"
CRYPTO_PORTFOLIO_EQUITY_PNG = "research_results/crypto_portfolio_equity.png"


def _load_config() -> BacktestConfig:
    """Load backtest config from env with defaults."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path)
    return BacktestConfig(
        symbol="ETHUSDT",
        interval=os.getenv("TIMEFRAME", "4h"),
        initial_capital=float(os.getenv("INITIAL_CAPITAL", "1000.0")),
        atr_multiplier=float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.0")),
        daily_loss_limit_percent=float(
            os.getenv("DAILY_LOSS_LIMIT_PERCENT", "3.0")
        ),
        risk_per_trade_percent=1.0,
        max_position_exposure_percent=50.0,
        lookback_years=int(os.getenv("BACKTEST_LOOKBACK_YEARS", "8")),
        trading_fee_percent=float(os.getenv("TRADING_FEE_PERCENT", "0.1")),
        slippage_percent=float(os.getenv("SLIPPAGE_PERCENT", "0.05")),
    )


def _fetch_ohlcv(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Fetch and normalize OHLCV for one symbol."""
    raw = fetch_historical_klines(symbol, interval, start_time, end_time)
    return normalize_ohlcv(raw)


def _position_size_from_risk_fraction(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_fraction: float,
    max_exposure_percent: float,
) -> float:
    """
    Position size when risk is given as a fraction of capital (e.g. from
    PortfolioRiskManager.adjust_position_size).
    """
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0
    price_risk = abs(entry_price - stop_loss)
    if price_risk == 0:
        return 0.0
    risk_amount = equity * risk_fraction
    position_size = risk_amount / price_risk
    position_value = position_size * entry_price
    max_position_value = equity * (max_exposure_percent / 100.0)
    if position_value > max_position_value:
        position_size = max_position_value / entry_price
    return position_size


def run_portfolio_backtest(
    config: Optional[BacktestConfig] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict], pd.Series]:
    """
    Run portfolio backtest across crypto assets with shared capital.

    Uses existing strategy signals; enforces 1% risk per trade and
    PortfolioRiskManager max 3% total portfolio risk. Tracks open
    positions per asset and produces portfolio-level metrics.

    Args:
        config: BacktestConfig; if None, loaded from env.
        verbose: If True, print progress and summary.

    Returns:
        (metrics_dict, equity_df, trades_list, equity_series).
    """
    if config is None:
        config = _load_config()

    end_time = datetime.utcnow() - timedelta(minutes=5)
    start_time = end_time - timedelta(days=365 * config.lookback_years)

    strategy = TradingStrategy(atr_multiplier=config.atr_multiplier)
    fee_pct = config.trading_fee_percent / 100.0
    slippage_pct = config.slippage_percent / 100.0

    # Fetch data and build indicator DataFrames per asset
    asset_dfs: Dict[str, pd.DataFrame] = {}
    for symbol in PORTFOLIO_ASSETS:
        if verbose:
            print(f"Running asset: {symbol}")
        try:
            ohlcv = _fetch_ohlcv(
                symbol, config.interval, start_time, end_time
            )
        except Exception as exc:
            if verbose:
                print(f"  Skip {symbol}: {exc}")
            continue
        if len(ohlcv) < 200:
            if verbose:
                print(f"  Skip {symbol}: insufficient bars ({len(ohlcv)})")
            continue
        df = strategy.calculate_indicators(ohlcv)
        df = df.sort_index()
        asset_dfs[symbol] = df

    if not asset_dfs:
        raise RuntimeError("No asset data available for portfolio backtest.")

    # Unified timeline: sorted union of all bar timestamps
    all_ts: List[pd.Timestamp] = []
    for df in asset_dfs.values():
        all_ts.extend(df.index.tolist())
    timeline = sorted(set(all_ts))

    initial_capital = config.initial_capital
    cash = initial_capital
    # Per-asset position: size, entry_price, stop_loss, side, fee_entry_paid, risk_amount
    positions: Dict[str, Dict[str, Any]] = {}
    risk_mgr = PortfolioRiskManager(initial_capital)
    trades: List[Dict] = []
    equity_curve: List[float] = []
    equity_index: List[pd.Timestamp] = []

    current_day: Optional[datetime.date] = None
    start_of_day_equity = initial_capital
    daily_pnl = 0.0
    daily_loss_limit_hit = False

    for t in timeline:
        # Mark-to-market: current price per asset is last close up to t
        equity = cash
        for sym, pos in positions.items():
            if sym not in asset_dfs:
                continue
            df_sym = asset_dfs[sym]
            available = df_sym.index[df_sym.index <= t]
            if len(available) == 0:
                continue
            last_close = float(df_sym.loc[available[-1], "close"])
            size = pos["size"]
            if pos["side"] == "short":
                equity -= size * last_close
            else:
                equity += size * last_close

        risk_mgr.set_capital(equity)

        # New day bookkeeping
        bar_day = t.date() if hasattr(t, "date") else t
        if current_day != bar_day:
            current_day = bar_day
            start_of_day_equity = equity
            daily_pnl = 0.0
            daily_loss_limit_hit = False

        # Process each asset that has a bar at t
        for symbol in list(asset_dfs.keys()):
            df_sym = asset_dfs[symbol]
            if t not in df_sym.index:
                continue
            row = df_sym.loc[t]
            price_close = float(row["close"])
            price_low = float(row["low"])
            price_high = float(row["high"])
            df_slice = df_sym.loc[:t]

            # --- Exit handling ---
            if symbol in positions:
                pos = positions[symbol]
                stop_loss = pos["stop_loss"]
                position_side = pos["side"]
                entry_price = pos["entry_price"]
                position_size = pos["size"]
                fee_entry_paid = pos["fee_entry_paid"]
                risk_amount = pos["risk_amount"]

                if position_side == "short":
                    stop_hit = price_high >= stop_loss
                    exit_signal, _ = strategy.check_short_exit_signal(
                        df_slice, entry_price
                    )
                else:
                    stop_hit = price_low <= stop_loss
                    exit_signal, _ = strategy.check_exit_signal(
                        df_slice, entry_price
                    )

                should_exit = stop_hit or exit_signal
                if should_exit:
                    exit_reason = "stop_loss" if stop_hit else "signal"
                    exit_price_raw = stop_loss if stop_hit else price_close
                    if position_side == "short":
                        effective_exit = exit_price_raw * (1.0 + slippage_pct)
                        fee_exit = position_size * effective_exit * fee_pct
                        cash -= position_size * effective_exit + fee_exit
                        trade_pnl = (
                            (entry_price - effective_exit) * position_size
                            - fee_exit
                            - fee_entry_paid
                        )
                    else:
                        effective_exit = exit_price_raw * (1.0 - slippage_pct)
                        fee_exit = position_size * effective_exit * fee_pct
                        cash += position_size * effective_exit - fee_exit
                        trade_pnl = (
                            (effective_exit - entry_price) * position_size
                            - fee_exit
                            - fee_entry_paid
                        )
                    trades.append({
                        "symbol": symbol,
                        "side": position_side,
                        "entry_time": pos["entry_time"],
                        "exit_time": t.isoformat(),
                        "entry_price": float(entry_price),
                        "exit_price": float(effective_exit),
                        "quantity": float(position_size),
                        "pnl": float(trade_pnl),
                        "return_pct": (
                            trade_pnl / start_of_day_equity * 100.0
                            if start_of_day_equity > 0
                            else 0.0
                        ),
                        "exit_reason": exit_reason,
                    })
                    daily_pnl += trade_pnl
                    risk_mgr.track_position_close(risk_amount)
                    del positions[symbol]
                    if start_of_day_equity > 0 and daily_pnl <= -(
                        config.daily_loss_limit_percent / 100.0
                        * start_of_day_equity
                    ):
                        daily_loss_limit_hit = True
                continue

            # --- Entry handling (no position in this asset) ---
            if daily_loss_limit_hit:
                continue

            long_ok, long_details = strategy.check_entry_signal(df_slice)
            short_ok, short_details = strategy.check_short_entry_signal(
                df_slice
            )
            signal_details = None
            side: Optional[str] = None
            if long_ok and long_details is not None:
                signal_details = long_details
                side = "long"
            elif short_ok and short_details is not None:
                signal_details = short_details
                side = "short"

            if signal_details is None or side is None:
                continue

            atr = float(signal_details["atr"])
            entry_price_signal = float(signal_details["entry_price"])
            stop_loss = float(
                strategy.calculate_stop_loss(
                    entry_price_signal, atr, side=side
                )
            )
            risk_amount_desired = equity * 0.01  # 1% of capital
            if not risk_mgr.can_enter_trade(risk_amount_desired):
                continue
            allowed_risk_fraction = risk_mgr.adjust_position_size(0.01)
            if allowed_risk_fraction <= 0:
                continue
            size = _position_size_from_risk_fraction(
                equity=equity,
                entry_price=entry_price_signal,
                stop_loss=stop_loss,
                risk_fraction=allowed_risk_fraction,
                max_exposure_percent=config.max_position_exposure_percent,
            )
            if size <= 0:
                continue
            risk_amount_used = equity * allowed_risk_fraction
            if side == "short":
                effective_entry = entry_price_signal * (1.0 - slippage_pct)
                fee_entry = size * effective_entry * fee_pct
                cash += size * effective_entry - fee_entry
            else:
                effective_entry = entry_price_signal * (1.0 + slippage_pct)
                fee_entry = size * effective_entry * fee_pct
                cash -= size * effective_entry + fee_entry

            positions[symbol] = {
                "size": size,
                "entry_price": effective_entry,
                "stop_loss": stop_loss,
                "side": side,
                "fee_entry_paid": fee_entry,
                "risk_amount": risk_amount_used,
                "entry_time": t.isoformat(),
            }
            risk_mgr.track_position_open(risk_amount_used)

        # Portfolio equity at bar close (after exits/entries this bar)
        equity = cash
        for sym, pos in positions.items():
            if sym not in asset_dfs:
                continue
            df_sym = asset_dfs[sym]
            available = df_sym.index[df_sym.index <= t]
            if len(available) == 0:
                continue
            last_close = float(df_sym.loc[available[-1], "close"])
            size = pos["size"]
            if pos["side"] == "short":
                equity -= size * last_close
            else:
                equity += size * last_close
        equity_curve.append(equity)
        equity_index.append(t)

    # Final equity (mark open positions to last close)
    final_equity = cash
    for sym, pos in positions.items():
        if sym not in asset_dfs:
            continue
        last_close = float(asset_dfs[sym].iloc[-1]["close"])
        if pos["side"] == "short":
            final_equity -= pos["size"] * last_close
        else:
            final_equity += pos["size"] * last_close

    metrics = compute_performance_metrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=trades,
        equity_curve=equity_curve,
    )

    equity_series = pd.Series(equity_curve, index=equity_index)
    equity_df = pd.DataFrame({"datetime": equity_series.index, "equity": equity_curve})
    equity_df.set_index("datetime", inplace=True)

    if verbose:
        _print_summary_table(metrics)
        equity_df.to_csv(PORTFOLIO_EQUITY_CSV)
        print(f"\nPortfolio equity curve saved to '{PORTFOLIO_EQUITY_CSV}'.")

    return metrics, equity_df, trades, equity_series


def _print_summary_table(metrics: Dict[str, float]) -> None:
    """Print portfolio backtest summary table to terminal."""
    print("\n--- Portfolio backtest summary ---")
    print(f"  total_return   : {metrics['total_return']:.2f}%")
    pf = metrics["profit_factor"]
    pf_str = "Infinity" if math.isinf(pf) else f"{pf:.2f}"
    print(f"  profit_factor  : {pf_str}")
    print(f"  max_drawdown   : {metrics['max_drawdown']:.2f}%")
    print(f"  sharpe_ratio   : {metrics['sharpe_ratio']:.2f}")
    print(f"  total_trades   : {int(metrics['total_trades'])}")


def save_crypto_portfolio_equity_chart(equity_series: pd.Series) -> None:
    """
    Plot portfolio equity curve and save to research_results/crypto_portfolio_equity.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(CRYPTO_PORTFOLIO_EQUITY_PNG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_series.index, equity_series.values, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio equity")
    ax.set_title("Crypto portfolio equity curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CRYPTO_PORTFOLIO_EQUITY_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    metrics, equity_df, trades, equity_series = run_portfolio_backtest(
        verbose=True
    )
    save_crypto_portfolio_equity_chart(equity_series)
    print(f"\nChart saved to '{CRYPTO_PORTFOLIO_EQUITY_PNG}'.")
