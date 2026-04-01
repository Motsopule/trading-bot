"""
Backtesting engine for the MA crossover crypto trading strategy.

This module:
- Downloads historical OHLCV data from Binance public REST API
- Calculates MA20, MA50, MA200, ATR(14), and ATR_MA(20)
- Applies the long/short MA crossover strategy:
  * Long entry:  50MA > 200MA, MA20 crosses above MA50, ATR(14) > ATR_MA(20)
  * Long exit:   MA20 crosses below MA50 OR close < MA50 OR long stop-loss hit
  * Short entry: 50MA < 200MA, MA20 crosses below MA50, ATR(14) > ATR_MA(20)
  * Short exit:  MA20 crosses above MA50 OR close > MA50 OR short stop-loss hit
- Enforces:
  * Daily loss limit: 3% of start-of-day equity (blocks new entries only)
  * Stop-loss: 2 × ATR(14) from entry (below for longs, above for shorts)
- Applies configurable trading fees and slippage to all entries and exits
- Simulates trades with simple position sizing rules
- Outputs performance statistics and a CSV trade log.

IMPORTANT:
    This is a pure backtest module. It never sends real orders and does not
    depend on the live trading execution layer.
"""

from __future__ import annotations

import calendar
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from research.data_providers.http_retry import http_get_with_retries
from strategy import TradingStrategy


BINANCE_BASE_URL = "https://api.binance.com"

# Pause after each successful klines request to reduce API rate pressure.
REQUEST_DELAY = 0.5


@dataclass
class BacktestConfig:
    """Configuration parameters for the backtest engine."""

    symbol: str = "ETHUSDT"
    interval: str = "4h"
    initial_capital: float = 1000.0
    atr_multiplier: float = 2.0
    daily_loss_limit_percent: float = 3.0
    risk_per_trade_percent: float = 1.0
    max_position_exposure_percent: float = 50.0
    lookback_years: int = 8  # Max available on Binance 4h (~8 years for BTC/ETH)
    trade_log_csv: str = "backtest_trades.csv"
    trading_fee_percent: float = 0.1  # 0.1% per side
    slippage_percent: float = 0.05   # 0.05%


def _utc_naive(dt: datetime) -> datetime:
    """Convert aware UTC or naive datetime to UTC-naive for calendar.timegm."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _interval_to_milliseconds(interval: str) -> int:
    """
    Convert Binance kline interval string to milliseconds.

    Args:
        interval: Interval string (e.g. '1m', '4h', '1d').

    Returns:
        Interval length in milliseconds.
    """
    unit = interval[-1]
    amount = int(interval[:-1])

    if unit == "m":
        return amount * 60 * 1000
    if unit == "h":
        return amount * 60 * 60 * 1000
    if unit == "d":
        return amount * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_historical_klines(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    max_per_request: int = 1000,
) -> pd.DataFrame:
    """
    Download historical klines from Binance public REST API.

    Args:
        symbol: Trading pair symbol (e.g. 'ETHUSDT').
        interval: Kline interval (e.g. '4h').
        start_time: Start of backtest window (UTC).
        end_time: End of backtest window (UTC).
        max_per_request: Maximum candles per API request (default: 1000).

    Returns:
        DataFrame with OHLCV data indexed by UTC datetime.
    """
    safe_end = datetime.utcnow() - timedelta(minutes=5)
    start_date = _utc_naive(start_time)
    end_naive = _utc_naive(end_time)
    end_date = min(end_naive, safe_end)
    if start_date >= end_date:
        raise ValueError(
            f"Invalid range after safe end cap: start={start_date} end={end_date}"
        )
    logging.info(f"[BINANCE SAFE RANGE] {symbol} → {start_date} to {end_date}")
    start_ms = int(calendar.timegm(start_date.timetuple()) * 1000)
    end_ms = int(calendar.timegm(end_date.timetuple()) * 1000)
    interval_ms = _interval_to_milliseconds(interval)

    all_klines: List[List] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": max_per_request,
        }
        response = http_get_with_retries(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params=params,
        )
        klines = response.json()
        time.sleep(REQUEST_DELAY)
        if not klines:
            break

        all_klines.extend(klines)

        # Advance start to just after last returned kline to avoid overlap
        last_open_time = klines[-1][0]
        current_start = last_open_time + interval_ms

        # Safety: stop if we somehow do not move forward
        if current_start <= last_open_time:
            break

    if not all_klines:
        raise RuntimeError("No kline data returned from Binance.")

    df = pd.DataFrame(
        all_klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df["datetime"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]]

    return df


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float,
    max_exposure_percent: float,
) -> float:
    """
    Calculate position size based on simple risk rules.

    This mirrors the live RiskManager logic:
    - Risk a fixed percentage of equity per trade.
    - Cap position exposure to a maximum percentage of equity.

    Args:
        equity: Current account equity.
        entry_price: Planned entry price.
        stop_loss: Stop-loss price.
        risk_percent: Percentage of equity to risk per trade.
        max_exposure_percent: Maximum position value as percentage of equity.

    Returns:
        Position size in base asset units.
    """
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0

    price_risk = abs(entry_price - stop_loss)
    if price_risk == 0:
        return 0.0

    risk_amount = equity * (risk_percent / 100.0)
    position_size = risk_amount / price_risk
    position_value = position_size * entry_price
    max_position_value = equity * (max_exposure_percent / 100.0)

    if position_value > max_position_value:
        position_size = max_position_value / entry_price

    return position_size


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """
    Compute maximum drawdown in percentage from an equity curve.

    Args:
        equity_curve: Sequence of equity values over time.

    Returns:
        Maximum drawdown as a negative percentage (e.g. -12.5 for -12.5%).
        Returns 0.0 if drawdown cannot be computed.
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100.0
        if dd < max_dd:
            max_dd = dd

    return max_dd


def compute_performance_metrics(
    initial_capital: float,
    final_equity: float,
    trades: List[Dict],
    equity_curve: List[float],
) -> Dict[str, float]:
    """
    Compute summary performance metrics from backtest results.

    Args:
        initial_capital: Starting capital.
        final_equity: Final equity after backtest.
        trades: List of trade dictionaries.
        equity_curve: Equity values over time (per bar).

    Returns:
        Dictionary of performance metrics.
    """
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]

    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    max_drawdown = compute_max_drawdown(equity_curve)
    total_return = (final_equity / initial_capital - 1.0) * 100.0

    # Annualized Sharpe from bar returns (4h bars: ~2190 per year)
    sharpe_ratio = 0.0
    if len(equity_curve) > 1 and equity_curve[0] > 0:
        eq_series = pd.Series(equity_curve)
        bar_returns = eq_series.pct_change().dropna()
        if len(bar_returns) > 0 and bar_returns.std() > 0:
            periods_per_year = 365 * 24 / 4  # 4h bars
            sharpe_ratio = float(
                bar_returns.mean() / bar_returns.std() * math.sqrt(periods_per_year)
            )

    return {
        "total_trades": float(total_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
    }


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an OHLCV DataFrame to standard columns (open, high, low, close, volume)
    and ensure a datetime index. Accepts yfinance-style (Open, High, etc.) or
    already lowercased columns.

    Args:
        df: Raw OHLCV DataFrame, possibly with MultiIndex columns from yfinance.

    Returns:
        DataFrame with columns open, high, low, close, volume and datetime index.
    """
    if df is None or df.empty:
        return df
    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    col_map = {}
    for c in df.columns:
        c_lower = str(c).lower()
        if c_lower in ("open", "high", "low", "close", "volume"):
            col_map[c] = c_lower
    df = df.rename(columns=col_map)
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"DataFrame must have open, high, low, close; got {list(df.columns)}"
        )
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def run_backtest_with_ohlcv(
    ohlcv: pd.DataFrame,
    config: BacktestConfig,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict, pd.Series]:
    """
    Run the backtest engine on pre-fetched OHLCV data (e.g. from yfinance).

    Strategy and risk logic are unchanged. Use this for non-Binance markets.

    Args:
        ohlcv: DataFrame with open, high, low, close, volume and datetime index.
        config: BacktestConfig (symbol used for trade log and labels).
        verbose: If True, print summary and save trade log CSV.

    Returns:
        Tuple of (trades_df, metrics_dict, equity_series with datetime index).
    """
    ohlcv = normalize_ohlcv(ohlcv.copy())
    if len(ohlcv) < 200:
        raise ValueError(
            f"Insufficient bars for strategy (need 200+, got {len(ohlcv)})"
        )

    strategy = TradingStrategy(atr_multiplier=config.atr_multiplier)
    df = strategy.calculate_indicators(ohlcv)

    initial_capital = config.initial_capital
    cash = initial_capital
    position_size = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    in_position = False
    position_side: Optional[str] = None
    fee_entry_paid: float = 0.0

    fee_pct = config.trading_fee_percent / 100.0
    slippage_pct = config.slippage_percent / 100.0

    equity_curve: List[float] = []
    trades: List[Dict] = []

    current_day: Optional[datetime.date] = None
    start_of_day_equity: float = initial_capital
    daily_pnl: float = 0.0
    daily_loss_limit_hit: bool = False

    df_sorted = df.sort_index()

    for i in range(1, len(df_sorted)):
        row = df_sorted.iloc[i]
        prev_row = df_sorted.iloc[i - 1]
        ts: pd.Timestamp = row.name  # index as timestamp
        price_close = float(row["close"])
        price_low = float(row["low"])
        price_high = float(row["high"])

        # Mark-to-market equity at bar close
        if in_position:
            if position_side == "short":
                equity = cash - position_size * price_close
            else:
                equity = cash + position_size * price_close
        else:
            equity = cash
        equity_curve.append(equity)

        # Handle new day bookkeeping
        bar_day = ts.date()
        if current_day != bar_day:
            current_day = bar_day
            start_of_day_equity = equity
            daily_pnl = 0.0
            daily_loss_limit_hit = False

        # Manage open position (check stop-loss and exit signal)
        if in_position:
            if position_side == "short":
                # For shorts, stop is above market; use high-of-bar for intrabar breaches
                stop_hit = price_high >= stop_loss
                exit_signal, _ = strategy.check_short_exit_signal(
                    df_sorted.iloc[: i + 1], entry_price
                )
            else:
                # Long: stop below market; use low-of-bar for intrabar breaches
                stop_hit = price_low <= stop_loss
                exit_signal, _ = strategy.check_exit_signal(
                    df_sorted.iloc[: i + 1], entry_price
                )

            should_exit = stop_hit or exit_signal
            if should_exit:
                exit_reason = "stop_loss" if stop_hit else "signal"
                exit_price_raw = stop_loss if stop_hit else price_close

                # Apply slippage and fees to exit
                if position_side == "short":
                    effective_exit = exit_price_raw * (1.0 + slippage_pct)
                    fee_exit = position_size * effective_exit * fee_pct
                    cash -= position_size * effective_exit + fee_exit
                    trade_pnl = (entry_price - effective_exit) * position_size - fee_exit - fee_entry_paid
                else:
                    effective_exit = exit_price_raw * (1.0 - slippage_pct)
                    fee_exit = position_size * effective_exit * fee_pct
                    cash += position_size * effective_exit - fee_exit
                    trade_pnl = (effective_exit - entry_price) * position_size - fee_exit - fee_entry_paid

                trade = {
                    "symbol": config.symbol,
                    "side": position_side,
                    "entry_time": prev_row.name.isoformat(),
                    "exit_time": ts.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_price": float(effective_exit),
                    "quantity": float(position_size),
                    "pnl": float(trade_pnl),
                    "return_pct": float(
                        trade_pnl / start_of_day_equity * 100.0
                        if start_of_day_equity > 0
                        else 0.0
                    ),
                    "exit_reason": exit_reason,
                }
                trades.append(trade)

                daily_pnl += trade_pnl
                in_position = False
                position_size = 0.0
                entry_price = 0.0
                stop_loss = 0.0
                position_side = None
                fee_entry_paid = 0.0

                # Enforce daily loss limit (3% of start-of-day equity)
                if start_of_day_equity > 0:
                    if daily_pnl <= -(
                        config.daily_loss_limit_percent / 100.0 * start_of_day_equity
                    ):
                        daily_loss_limit_hit = True

        # Potential new entry on this bar (long or short). Crypto trades 24/7,
        # so we evaluate the strategy on every 4H candle and do not restrict
        # entries to a specific intraday session.
        if not in_position and not daily_loss_limit_hit:
            long_entry_signal, long_details = strategy.check_entry_signal(
                df_sorted.iloc[: i + 1]
            )
            short_entry_signal, short_details = strategy.check_short_entry_signal(
                df_sorted.iloc[: i + 1]
            )

            signal_details = None
            side: Optional[str] = None
            if long_entry_signal and long_details is not None:
                signal_details = long_details
                side = "long"
            elif short_entry_signal and short_details is not None:
                signal_details = short_details
                side = "short"

            if signal_details is not None and side is not None:
                atr = float(signal_details["atr"])
                entry_price_signal = float(signal_details["entry_price"])
                stop_loss = float(
                    strategy.calculate_stop_loss(
                        entry_price_signal, atr, side=side
                    )
                )

                size = calculate_position_size(
                    equity=equity,
                    entry_price=entry_price_signal,
                    stop_loss=stop_loss,
                    risk_percent=config.risk_per_trade_percent,
                    max_exposure_percent=config.max_position_exposure_percent,
                )

                if size > 0:
                    # Apply slippage to execution price
                    if side == "short":
                        effective_entry = entry_price_signal * (
                            1.0 - slippage_pct
                        )
                        fee_entry = size * effective_entry * fee_pct
                        cash += size * effective_entry - fee_entry
                    else:
                        effective_entry = entry_price_signal * (
                            1.0 + slippage_pct
                        )
                        fee_entry = size * effective_entry * fee_pct
                        cash -= size * effective_entry + fee_entry

                    position_size = size
                    position_side = side
                    entry_price = effective_entry
                    fee_entry_paid = fee_entry
                    in_position = True

    last_close = float(df_sorted.iloc[-1]["close"])
    if in_position and position_side == "short":
        final_equity = cash - position_size * last_close
    else:
        final_equity = cash + position_size * last_close
    metrics = compute_performance_metrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=trades,
        equity_curve=equity_curve,
    )

    trades_df = pd.DataFrame(trades)
    equity_index = df_sorted.index[1:]
    equity_series = pd.Series(equity_curve, index=equity_index)

    if verbose:
        if not trades_df.empty:
            trades_df.to_csv(config.trade_log_csv, index=False)
        print("Backtest results:")
        print(f"  Total trades   : {int(metrics['total_trades'])}")
        print(f"  Win rate       : {metrics['win_rate']:.2f}%")
        if math.isinf(metrics["profit_factor"]):
            pf_str = "Infinity"
        else:
            pf_str = f"{metrics['profit_factor']:.2f}"
        print(f"  Profit factor  : {pf_str}")
        print(f"  Max drawdown   : {metrics['max_drawdown']:.2f}%")
        print(f"  Total return   : {metrics['total_return']:.2f}%")
        if not trades_df.empty:
            print(f"Trade log saved to '{config.trade_log_csv}'.")
        else:
            print("No trades generated; trade log not created.")

    return trades_df, metrics, equity_series


def run_backtest(config: Optional[BacktestConfig] = None) -> Tuple[pd.DataFrame, Dict, pd.Series]:
    """
    Run the backtest and return trade log, performance metrics, and equity curve.

    Fetches historical data from Binance, then runs the same engine as
    run_backtest_with_ohlcv.

    Args:
        config: Optional BacktestConfig; if None, values are loaded from the
            environment (falling back to sensible defaults).

    Returns:
        Tuple of (trades_df, metrics_dict, equity_series).
    """
    # Load environment so user can override defaults via .env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path)

    if config is None:
        config = BacktestConfig(
            symbol=os.getenv("TRADING_PAIR", "ETHUSDT"),
            interval=os.getenv("TIMEFRAME", "4h"),
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "1000.0")),
            atr_multiplier=float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.0")),
            daily_loss_limit_percent=float(
                os.getenv("DAILY_LOSS_LIMIT_PERCENT", "3.0")
            ),
            lookback_years=int(os.getenv("BACKTEST_LOOKBACK_YEARS", "8")),
            trading_fee_percent=float(os.getenv("TRADING_FEE_PERCENT", "0.1")),
            slippage_percent=float(os.getenv("SLIPPAGE_PERCENT", "0.05")),
        )

    end_time = datetime.utcnow() - timedelta(minutes=5)
    start_time = end_time - timedelta(days=365 * config.lookback_years)

    print(
        f"Fetching data for {config.symbol} {config.interval} from "
        f"{start_time.date()} to {end_time.date()}..."
    )
    ohlcv = fetch_historical_klines(
        symbol=config.symbol,
        interval=config.interval,
        start_time=start_time,
        end_time=end_time,
    )

    return run_backtest_with_ohlcv(ohlcv, config, verbose=True)


if __name__ == "__main__":
    run_backtest()

