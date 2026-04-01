"""
Cross-asset backtest for the mean reversion research strategy.

Runs the strategy on crypto (Binance), forex, indices, and commodities (yfinance).
Uses the same OHLCV normalization approach as previous research scripts.
All code is research-only; production files are not modified.
"""

from __future__ import annotations

import calendar
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path when running as script (research/mean_reversion_backtest.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from research.constants import LOOKBACK_TRADITIONAL_YEARS
from research.data_providers.http_retry import (
    http_get_with_retries,
    yfinance_download_with_retries,
)
from research.mean_reversion_strategy import (
    ATR_STOP_MULT,
    RISK_PER_TRADE_PERCENT,
    calculate_indicators,
    calculate_stop_loss,
    generate_signal,
)

# Timeframe for all backtests (4h)
TIMEFRAME = "4h"
BINANCE_BASE_URL = "https://api.binance.com"

# Crypto pairs (Binance)
CRYPTO_PAIRS: List[str] = [
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "BTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "ADAUSDT",
]

# Forex: (display symbol, yfinance ticker)
FOREX: List[Tuple[str, str]] = [
    ("EURUSD", "EURUSD=X"),
    ("GBPUSD", "GBPUSD=X"),
    ("USDJPY", "USDJPY=X"),
]

# Indices: (display symbol, yfinance ticker)
INDICES: List[Tuple[str, str]] = [
    ("SP500", "^GSPC"),
    ("NASDAQ", "^IXIC"),
]

# Commodities: (display symbol, yfinance ticker)
COMMODITIES: List[Tuple[str, str]] = [
    ("GOLD", "GC=F"),
    ("CRUDE_OIL", "CL=F"),
]

# Backtest config defaults
INITIAL_CAPITAL = 1000.0
TRADING_FEE_PERCENT = 0.1
SLIPPAGE_PERCENT = 0.05
MAX_POSITION_EXPOSURE_PERCENT = 50.0

_log = logging.getLogger(__name__)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV to columns open, high, low, close, volume and datetime index.

    Same approach as in backtest.py / previous research scripts.
    """
    if df is None or df.empty:
        return df
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


def _interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60 * 1000
    if unit == "h":
        return amount * 60 * 60 * 1000
    if unit == "d":
        return amount * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_binance_klines(
    symbol: str,
    interval: str,
    lookback_years: int = 8,
) -> pd.DataFrame:
    """Fetch 4h klines from Binance (same as production backtest data source)."""
    end_date = datetime.utcnow() - timedelta(minutes=5)
    start_date = end_date - timedelta(days=365 * lookback_years)
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
            "limit": 1000,
        }
        response = http_get_with_retries(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params=params,
        )
        klines = response.json()
        if not klines:
            break
        all_klines.extend(klines)
        last_open_time = klines[-1][0]
        current_start = last_open_time + interval_ms
        if current_start <= last_open_time:
            break
    if not all_klines:
        raise RuntimeError("No kline data returned from Binance.")
    df = pd.DataFrame(
        all_klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def _daily_to_pseudo_4h(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily OHLCV into 6 pseudo 4h bars per day (same as multi_backtest)."""
    if daily_df.empty:
        return daily_df
    hours = [0, 4, 8, 12, 16, 20]
    rows = []
    for ts, row in daily_df.iterrows():
        if hasattr(ts, "tz") and ts.tz is not None:
            base = ts
        else:
            base = pd.Timestamp(ts).tz_localize("UTC", ambiguous="raise")
        date_part = base.date() if hasattr(base, "date") else base.normalize().date()
        for h in hours:
            bar_ts = pd.Timestamp(
                date_part.year, date_part.month, date_part.day, h, 0, 0, tz="UTC"
            )
            rows.append({
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"] / 6.0,
                "timestamp": bar_ts,
            })
    out = pd.DataFrame(rows).set_index("timestamp")
    out.index.name = None
    return out.sort_index()


def fetch_yfinance_4h(
    yf_ticker: str,
    lookback_years: int = LOOKBACK_TRADITIONAL_YEARS,
) -> pd.DataFrame:
    """Fetch OHLCV from yfinance and return 4h bars (same approach as multi_backtest)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * lookback_years)
    use_daily_fallback = lookback_years > 2
    if use_daily_fallback:
        dl = yfinance_download_with_retries(
            yf_ticker, start=start, end=end,
            interval="1d", progress=False, auto_adjust=True,
        )
    else:
        dl = yfinance_download_with_retries(
            yf_ticker, start=start, end=end,
            interval="1h", progress=False, auto_adjust=True,
        )
    if dl is None or dl.empty or len(dl) < 200:
        raise ValueError(
            f"Insufficient yfinance data for {yf_ticker}: "
            f"got {len(dl) if dl is not None else 0} bars"
        )
    if isinstance(dl.columns, pd.MultiIndex):
        dl.columns = dl.columns.get_level_values(0)
    for c in list(dl.columns):
        c_lower = str(c).lower()
        if c_lower in ("open", "high", "low", "close", "volume"):
            dl = dl.rename(columns={c: c_lower})
    if "volume" not in dl.columns:
        dl["volume"] = 0.0
    dl = dl[["open", "high", "low", "close", "volume"]].dropna()
    if use_daily_fallback:
        dl = _daily_to_pseudo_4h(dl)
    else:
        if dl.index.inferred_freq != "D" and len(dl) > 4 * 200:
            resampled = dl.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna(how="all")
            resampled["volume"] = resampled["volume"].fillna(0)
            dl = resampled.dropna(subset=["open", "high", "low", "close"])
    if dl.index.tz is None:
        dl.index = dl.index.tz_localize("UTC", ambiguous="raise")
    return dl


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = RISK_PER_TRADE_PERCENT,
    max_exposure_percent: float = MAX_POSITION_EXPOSURE_PERCENT,
) -> float:
    """Position size so that risk is risk_percent of equity (1% default)."""
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
    """Max drawdown in percentage (negative)."""
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
    """Compute total_trades, win_rate, profit_factor, max_drawdown, sharpe_ratio, total_return."""
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
    sharpe_ratio = 0.0
    if len(equity_curve) > 1 and equity_curve[0] > 0:
        eq_series = pd.Series(equity_curve)
        bar_returns = eq_series.pct_change().dropna()
        if len(bar_returns) > 0 and bar_returns.std() > 0:
            periods_per_year = 365 * 24 / 4
            sharpe_ratio = float(
                bar_returns.mean() / bar_returns.std() * math.sqrt(periods_per_year)
            )
    return {
        "total_trades": float(total_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
    }


def run_single_backtest(
    ohlcv: pd.DataFrame,
    symbol: str,
    initial_capital: float = INITIAL_CAPITAL,
    fee_pct: float = TRADING_FEE_PERCENT / 100.0,
    slippage_pct: float = SLIPPAGE_PERCENT / 100.0,
) -> Tuple[Dict[str, float], pd.Series, List[Dict]]:
    """
    Run mean reversion backtest on one OHLCV series.

    Returns:
        (metrics_dict, equity_series, trades_list).
    """
    ohlcv = normalize_ohlcv(ohlcv.copy())
    if len(ohlcv) < 200:
        raise ValueError(f"Insufficient bars (need 200+, got {len(ohlcv)})")

    df = calculate_indicators(ohlcv)
    df = generate_signal(df)
    df = df.dropna(subset=["ma50", "atr_14", "rsi_14", "long_entry", "short_entry"])
    df_sorted = df.sort_index()

    cash = initial_capital
    position_size = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    atr_at_entry = 0.0
    in_position = False
    position_side: Optional[str] = None
    fee_entry_paid = 0.0

    equity_curve: List[float] = []
    trades: List[Dict] = []

    for i in range(1, len(df_sorted)):
        row = df_sorted.iloc[i]
        prev_row = df_sorted.iloc[i - 1]
        ts = row.name
        price_close = float(row["close"])
        price_low = float(row["low"])
        price_high = float(row["high"])

        if in_position:
            if position_side == "short":
                equity = cash - position_size * price_close
            else:
                equity = cash + position_size * price_close
        else:
            equity = cash
        equity_curve.append(equity)

        # Exit logic: stop (fixed at entry) or signal
        if in_position:
            if position_side == "long":
                stop_hit = price_low <= stop_loss
                exit_signal = bool(row["long_exit"])
            else:
                stop_hit = price_high >= stop_loss
                exit_signal = bool(row["short_exit"])

            if stop_hit or exit_signal:
                exit_price_raw = stop_loss if stop_hit else price_close
                if position_side == "short":
                    effective_exit = exit_price_raw * (1.0 + slippage_pct)
                    fee_exit = position_size * effective_exit * fee_pct
                    cash -= position_size * effective_exit + fee_exit
                    trade_pnl = (
                        (entry_price - effective_exit) * position_size
                        - fee_exit - fee_entry_paid
                    )
                else:
                    effective_exit = exit_price_raw * (1.0 - slippage_pct)
                    fee_exit = position_size * effective_exit * fee_pct
                    cash += position_size * effective_exit - fee_exit
                    trade_pnl = (
                        (effective_exit - entry_price) * position_size
                        - fee_exit - fee_entry_paid
                    )
                trades.append({
                    "symbol": symbol,
                    "side": position_side,
                    "entry_price": float(entry_price),
                    "exit_price": float(effective_exit),
                    "quantity": float(position_size),
                    "pnl": float(trade_pnl),
                })
                in_position = False
                position_size = 0.0
                entry_price = 0.0
                stop_loss = 0.0
                position_side = None
                fee_entry_paid = 0.0

        # Entry logic
        if not in_position:
            side = None
            entry_price_signal = price_close
            atr_val = float(row["atr_14"])
            if bool(row["long_entry"]):
                side = "long"
            elif bool(row["short_entry"]):
                side = "short"

            if side is not None and atr_val > 0:
                stop_loss = calculate_stop_loss(entry_price_signal, atr_val, side=side)
                size = calculate_position_size(
                    equity=equity,
                    entry_price=entry_price_signal,
                    stop_loss=stop_loss,
                )
                if size > 0:
                    if side == "short":
                        effective_entry = entry_price_signal * (1.0 - slippage_pct)
                        fee_entry = size * effective_entry * fee_pct
                        cash += size * effective_entry - fee_entry
                    else:
                        effective_entry = entry_price_signal * (1.0 + slippage_pct)
                        fee_entry = size * effective_entry * fee_pct
                        cash -= size * effective_entry + fee_entry
                    position_size = size
                    position_side = side
                    entry_price = effective_entry
                    atr_at_entry = atr_val
                    fee_entry_paid = fee_entry
                    in_position = True

    last_close = float(df_sorted.iloc[-1]["close"])
    if in_position:
        if position_side == "short":
            final_equity = cash - position_size * last_close
        else:
            final_equity = cash + position_size * last_close
    else:
        final_equity = cash

    metrics = compute_performance_metrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=trades,
        equity_curve=equity_curve,
    )
    equity_index = df_sorted.index[1:]
    equity_series = pd.Series(equity_curve, index=equity_index)
    return metrics, equity_series, trades


def run_mean_reversion_backtest(
    lookback_crypto: int = 8,
    lookback_traditional: int = LOOKBACK_TRADITIONAL_YEARS,
    output_csv: str = "mean_reversion_results.csv",
    output_chart: str = "mean_reversion_equity_curves.png",
    results_dir: str = "research_results",
) -> pd.DataFrame:
    """
    Run mean reversion backtest across all assets and save results + equity chart.

    Prints "Running asset: X" for each asset. Saves CSV and PNG in results_dir.
    """
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_PROJECT_ROOT, os.path.normpath(results_dir))
    results_dir = os.path.normpath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, output_csv)
    chart_path = os.path.join(results_dir, output_chart)

    results: List[Dict] = []
    equity_curves: Dict[str, pd.Series] = {}

    # Crypto
    for pair in CRYPTO_PAIRS:
        print(f"Running asset: {pair}")
        try:
            ohlcv = fetch_binance_klines(pair, TIMEFRAME, lookback_years=lookback_crypto)
            metrics, equity_series, _ = run_single_backtest(ohlcv, pair)
            row = {"asset": pair, **metrics}
            results.append(row)
            equity_curves[pair] = equity_series
        except Exception as e:
            print(f"  Failed: {e}")

    _log.info(
        "Using lookback period: %s years for traditional assets",
        lookback_traditional,
    )

    # Forex
    for display_symbol, yf_ticker in FOREX:
        print(f"Running asset: {display_symbol}")
        try:
            ohlcv = fetch_yfinance_4h(yf_ticker, lookback_years=lookback_traditional)
            metrics, equity_series, _ = run_single_backtest(ohlcv, display_symbol)
            row = {"asset": display_symbol, **metrics}
            results.append(row)
            equity_curves[display_symbol] = equity_series
        except Exception as e:
            print(f"  Failed: {e}")

    # Indices
    for display_symbol, yf_ticker in INDICES:
        print(f"Running asset: {display_symbol}")
        try:
            ohlcv = fetch_yfinance_4h(yf_ticker, lookback_years=lookback_traditional)
            metrics, equity_series, _ = run_single_backtest(ohlcv, display_symbol)
            row = {"asset": display_symbol, **metrics}
            results.append(row)
            equity_curves[display_symbol] = equity_series
        except Exception as e:
            print(f"  Failed: {e}")

    # Commodities
    for display_symbol, yf_ticker in COMMODITIES:
        print(f"Running asset: {display_symbol}")
        try:
            ohlcv = fetch_yfinance_4h(yf_ticker, lookback_years=lookback_traditional)
            metrics, equity_series, _ = run_single_backtest(ohlcv, display_symbol)
            row = {"asset": display_symbol, **metrics}
            results.append(row)
            equity_curves[display_symbol] = equity_series
        except Exception as e:
            print(f"  Failed: {e}")

    if not results:
        print("No backtest results to save.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to '{csv_path}'.")

    # Equity curves chart
    if equity_curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for name, eq in equity_curves.items():
            if eq.empty or eq.iloc[0] == 0:
                continue
            norm = eq / eq.iloc[0] * 100.0
            ax.plot(norm.index, norm.values, label=name, alpha=0.8)
        ax.set_title("Mean reversion strategy — equity curves per asset (normalized to 100)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (index)")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Equity curves chart saved to '{chart_path}'.")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_mean_reversion_backtest(
        lookback_crypto=8,
        lookback_traditional=LOOKBACK_TRADITIONAL_YEARS,
        output_csv="mean_reversion_results.csv",
        output_chart="mean_reversion_equity_curves.png",
        results_dir="research_results",
    )
