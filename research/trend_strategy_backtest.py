"""
Backtest the production MA crossover trend strategy across multiple assets and timeframes.

Research-only: does not modify strategy.py, risk.py, execution.py, or main.py.
Implements the same logic as production (MA50 > MA200, MA20/50 crossover, 2×ATR stop, 1% risk)
for backtesting across 15m, 1h, 4h, 1d and crypto (Binance) + traditional assets
(Massive forex majors, yfinance for indices/commodities).
"""

from __future__ import annotations

DEBUG_MODE = True

import calendar
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path when running as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
import pandas as pd
from dotenv import load_dotenv
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from research.constants import LOOKBACK_TRADITIONAL_YEARS
from research.data_providers.http_retry import (
    http_get_with_retries,
    yfinance_download_with_retries,
)
from research.data_providers.massive import (
    FOREX_MASSIVE_SYMBOLS,
    fetch_massive_forex_ohlcv,
)
from research.data_providers.utc_range import utc_end_and_start

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_TIMEFRAMES_FULL = ["15m", "1h", "4h", "1d"]

# Crypto (Binance) — full list when DEBUG_MODE is False
_CRYPTOS_FULL: List[str] = [
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "AVAXUSDT",
    "ADAUSDT",
    "LINKUSDT",
]

# Traditional: (display symbol, yfinance ticker)
_FOREX_FULL: List[Tuple[str, str]] = [
    ("EURUSD", "EURUSD=X"),
    ("GBPUSD", "GBPUSD=X"),
    ("USDJPY", "USDJPY=X"),
]
_INDICES_FULL: List[Tuple[str, str]] = [
    ("SP500", "^GSPC"),
    ("NASDAQ", "^IXIC"),
]
_COMMODITIES_FULL: List[Tuple[str, str]] = [
    ("GOLD", "GC=F"),
    ("CRUDE_OIL", "CL=F"),
]

if DEBUG_MODE:
    ASSETS = {
        "crypto": ["ETHUSDT", "BTCUSDT", "SOLUSDT"],
        "forex": ["EURUSD", "GBPUSD"],
    }
    TIMEFRAMES = ["4h"]
    CRYPTO_ASSETS = list(ASSETS["crypto"])
    FOREX = [
        ("EURUSD", "EURUSD=X"),
        ("GBPUSD", "GBPUSD=X"),
    ]
    INDICES: List[Tuple[str, str]] = []
    COMMODITIES: List[Tuple[str, str]] = []
else:
    CRYPTO_ASSETS = _CRYPTOS_FULL
    FOREX = _FOREX_FULL
    INDICES = _INDICES_FULL
    COMMODITIES = _COMMODITIES_FULL
    TIMEFRAMES = _TIMEFRAMES_FULL
    ASSETS = {
        "crypto": list(CRYPTO_ASSETS),
        "forex": [d for d, _ in FOREX],
        "indices": [d for d, _ in INDICES],
        "commodities": [d for d, _ in COMMODITIES],
    }

BINANCE_BASE_URL = "https://api.binance.com"
LOOKBACK_CRYPTO_YEARS = 8
MIN_BARS = 200
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE_PERCENT = 1.0
ATR_MULTIPLIER = 2.0
TRADING_FEE_PERCENT = 0.1
SLIPPAGE_PERCENT = 0.05
MAX_POSITION_EXPOSURE_PERCENT = 50.0
MAX_PROFIT_FACTOR = 999.0

# Pause after successful API work to reduce rate pressure on multi-asset runs.
REQUEST_DELAY = 0.5

_log = logging.getLogger(__name__)
_traditional_lookback_logged = False


def _periods_per_year(timeframe: str) -> float:
    """
    Return number of bars per year for the given timeframe (for Sharpe).

    Args:
        timeframe: One of "15m", "1h", "4h", "1d".

    Returns:
        Bars per year.
    """
    if timeframe == "15m":
        return 365.0 * 24 * (60 // 15)
    if timeframe == "1h":
        return 365.0 * 24
    if timeframe == "4h":
        return 365.0 * (24 // 4)
    if timeframe == "1d":
        return 365.0
    raise ValueError(f"Unknown timeframe: {timeframe}")


def _interval_to_milliseconds(interval: str) -> int:
    """Convert interval string (e.g. '4h') to milliseconds."""
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60 * 1000
    if unit == "h":
        return amount * 60 * 60 * 1000
    if unit == "d":
        return amount * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV to columns open, high, low, close, volume and datetime index.

    Same approach as in other research scripts.
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


def fetch_binance_ohlcv(
    symbol: str,
    timeframe: str,
    lookback_years: int = LOOKBACK_CRYPTO_YEARS,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance for the given symbol and timeframe.

    Args:
        symbol: Binance symbol (e.g. ETHUSDT).
        timeframe: One of "15m", "1h", "4h", "1d".
        lookback_years: Years of history.

    Returns:
        DataFrame with open, high, low, close, volume and datetime index.
    """
    end_date = datetime.utcnow() - timedelta(minutes=5)
    start_date = end_date - timedelta(days=365 * lookback_years)
    _log.info(f"[BINANCE SAFE RANGE] {symbol} → {start_date} to {end_date}")
    end_ms = int(calendar.timegm(end_date.timetuple()) * 1000)
    start_ms = int(calendar.timegm(start_date.timetuple()) * 1000)
    interval_ms = _interval_to_milliseconds(timeframe)
    all_klines: List[List] = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
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
    return df[["open", "high", "low", "close", "volume"]]


def _daily_to_pseudo_bars(
    daily_df: pd.DataFrame,
    bars_per_day: int,
) -> pd.DataFrame:
    """
    Convert daily OHLCV into N pseudo intraday bars per day.

    Args:
        daily_df: DataFrame with open, high, low, close, volume.
        bars_per_day: 6 for 4h, 24 for 1h, 96 for 15m.

    Returns:
        DataFrame with datetime index and bars_per_day rows per day.
    """
    if daily_df.empty:
        return daily_df
    if bars_per_day == 1:
        return daily_df.copy()
    rows: List[Dict[str, Any]] = []
    interval_minutes = 24 * 60 // bars_per_day
    for ts, row in daily_df.iterrows():
        if hasattr(ts, "tz") and ts.tz is not None:
            base = ts
        else:
            base = pd.Timestamp(ts).tz_localize("UTC", ambiguous="raise")
        date_part = base.date() if hasattr(base, "date") else base.normalize().date()
        for i in range(bars_per_day):
            total_min = i * interval_minutes
            h = total_min // 60
            m = total_min % 60
            bar_ts = pd.Timestamp(
                date_part.year, date_part.month, date_part.day, h, m, 0, tz="UTC"
            )
            rows.append({
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"] / bars_per_day,
                "timestamp": bar_ts,
            })
    out = pd.DataFrame(rows).set_index("timestamp")
    out.index.name = None
    return out.sort_index()


def fetch_yfinance_ohlcv(
    yf_ticker: str,
    timeframe: str,
    lookback_years: int = LOOKBACK_TRADITIONAL_YEARS,
) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance and return bars at the requested timeframe.

    Args:
        yf_ticker: yfinance symbol (e.g. EURUSD=X, ^GSPC).
        timeframe: One of "15m", "1h", "4h", "1d".
        lookback_years: Years of history.

    Returns:
        DataFrame with open, high, low, close, volume and datetime index.
    """
    end_date, start_date = utc_end_and_start(lookback_years)
    if end_date > datetime.utcnow():
        raise ValueError(f"Invalid end_date detected: {end_date}")
    _log.info(
        f"[FINAL RANGE] {yf_ticker} {timeframe} → {start_date} to {end_date}"
    )
    end = end_date.replace(tzinfo=timezone.utc)
    start = start_date.replace(tzinfo=timezone.utc)
    use_daily = lookback_years > 2
    if use_daily:
        dl = yfinance_download_with_retries(
            yf_ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    else:
        dl = yfinance_download_with_retries(
            yf_ticker,
            start=start,
            end=end,
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
    if dl is None or dl.empty or len(dl) < 200:
        raise ValueError(
            f"Insufficient yfinance data for {yf_ticker}: "
            f"got {len(dl) if dl is not None else 0} bars"
        )
    if isinstance(dl.columns, pd.MultiIndex):
        dl.columns = dl.columns.get_level_values(0)
    col_map = {}
    for c in dl.columns:
        c_lower = str(c).lower()
        if c_lower in ("open", "high", "low", "close", "volume"):
            col_map[c] = c_lower
    dl = dl.rename(columns=col_map)
    if "volume" not in dl.columns:
        dl["volume"] = 0.0
    dl = dl[["open", "high", "low", "close", "volume"]].dropna()

    if timeframe == "1d":
        out = dl.copy()
    elif use_daily:
        if timeframe == "4h":
            bars_per_day = 6
        elif timeframe == "1h":
            bars_per_day = 24
        elif timeframe == "15m":
            bars_per_day = 96
        else:
            bars_per_day = 6
        out = _daily_to_pseudo_bars(dl, bars_per_day)
    else:
        if timeframe == "4h" and len(dl) > 4 * 200:
            out = dl.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(how="all")
            out["volume"] = out["volume"].fillna(0)
            out = out.dropna(subset=["open", "high", "low", "close"])
        else:
            out = dl
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC", ambiguous="raise")
    return out


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MA20, MA50, MA200 and ATR(14) for the trend strategy.

    Args:
        df: DataFrame with open, high, low, close, volume.

    Returns:
        DataFrame with added columns ma_20, ma_50, ma_200, atr.
    """
    if df is None or len(df) < 200:
        return df
    df = df.copy()
    df["ma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["ma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["ma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()
    return df


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = RISK_PER_TRADE_PERCENT,
    max_exposure_percent: float = MAX_POSITION_EXPOSURE_PERCENT,
) -> float:
    """
    Position size so that risk is risk_percent of equity (1% default).

    risk_amount = equity * (risk_percent/100); position_size = risk_amount / stop_distance.
    """
    if entry_price <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return 0.0
    risk_amount = equity * (risk_percent / 100.0)
    position_size = risk_amount / stop_distance
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
    periods_per_year: float,
) -> Dict[str, float]:
    """
    Compute total_trades, win_rate, profit_factor, max_drawdown, sharpe_ratio, total_return.

    Args:
        initial_capital: Starting equity.
        final_equity: Ending equity.
        trades: List of trade dicts with "pnl" key.
        equity_curve: List of equity values per bar.
        periods_per_year: Bars per year for Sharpe annualization.

    Returns:
        Dict with metric names and values.
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
    sharpe_ratio = 0.0
    if len(equity_curve) > 1 and equity_curve[0] > 0:
        eq_series = pd.Series(equity_curve)
        bar_returns = eq_series.pct_change().dropna()
        if len(bar_returns) > 0 and bar_returns.std() > 0:
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
    asset: str,
    timeframe: str,
    initial_capital: float = INITIAL_CAPITAL,
    fee_pct: float = TRADING_FEE_PERCENT / 100.0,
    slippage_pct: float = SLIPPAGE_PERCENT / 100.0,
) -> Tuple[Dict[str, float], pd.Series, List[Dict]]:
    """
    Run MA crossover trend strategy backtest (long only).

    Entry: MA50 > MA200 and MA20 crosses above MA50.
    Exit: MA20 crosses below MA50.
    Stop loss: entry - 2×ATR.
    Risk per trade: 1%.

    Args:
        ohlcv: OHLCV DataFrame.
        asset: Display name of the asset.
        timeframe: "15m", "1h", "4h", or "1d" (for Sharpe annualization).
        initial_capital: Starting equity.
        fee_pct: Fee per trade as decimal.
        slippage_pct: Slippage as decimal.

    Returns:
        (metrics_dict, equity_series, trades_list).
    """
    ohlcv = normalize_ohlcv(ohlcv.copy())
    if len(ohlcv) < MIN_BARS:
        raise ValueError(f"Insufficient bars (need {MIN_BARS}+, got {len(ohlcv)})")

    df = calculate_indicators(ohlcv)
    df = df.dropna(subset=["ma_20", "ma_50", "ma_200", "atr"])
    df_sorted = df.sort_index()

    cash = initial_capital
    position_size = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    in_position = False
    fee_entry_paid = 0.0

    equity_curve: List[float] = []
    trades: List[Dict] = []

    for i in range(1, len(df_sorted)):
        row = df_sorted.iloc[i]
        prev = df_sorted.iloc[i - 1]
        price_close = float(row["close"])
        price_low = float(row["low"])
        price_high = float(row["high"])
        ma20_curr = float(row["ma_20"])
        ma50_curr = float(row["ma_50"])
        ma200_curr = float(row["ma_200"])
        ma20_prev = float(prev["ma_20"])
        ma50_prev = float(prev["ma_50"])
        atr_val = float(row["atr"])

        if in_position:
            equity = cash + position_size * price_close
        else:
            equity = cash
        equity_curve.append(equity)

        # Exit: MA20 crosses below MA50 or stop hit
        if in_position:
            stop_hit = price_low <= stop_loss
            ma_cross_below = ma20_prev >= ma50_prev and ma20_curr < ma50_curr
            if stop_hit or ma_cross_below:
                exit_price_raw = stop_loss if stop_hit else price_close
                effective_exit = exit_price_raw * (1.0 - slippage_pct)
                fee_exit = position_size * effective_exit * fee_pct
                cash += position_size * effective_exit - fee_exit
                trade_pnl = (
                    (effective_exit - entry_price) * position_size
                    - fee_exit - fee_entry_paid
                )
                trades.append({
                    "symbol": asset,
                    "side": "long",
                    "entry_price": float(entry_price),
                    "exit_price": float(effective_exit),
                    "quantity": float(position_size),
                    "pnl": float(trade_pnl),
                })
                in_position = False
                position_size = 0.0
                entry_price = 0.0
                stop_loss = 0.0
                fee_entry_paid = 0.0

        # Entry: MA50 > MA200 and MA20 crosses above MA50
        if not in_position:
            trend_ok = ma50_curr > ma200_curr
            ma_cross_above = ma20_prev <= ma50_prev and ma20_curr > ma50_curr
            if trend_ok and ma_cross_above and atr_val > 0:
                stop_loss = price_close - ATR_MULTIPLIER * atr_val
                stop_loss = max(0.0, stop_loss)
                size = calculate_position_size(
                    equity=equity,
                    entry_price=price_close,
                    stop_loss=stop_loss,
                )
                if size > 0:
                    effective_entry = price_close * (1.0 + slippage_pct)
                    fee_entry = size * effective_entry * fee_pct
                    cash -= size * effective_entry + fee_entry
                    position_size = size
                    entry_price = effective_entry
                    fee_entry_paid = fee_entry
                    in_position = True

    last_close = float(df_sorted.iloc[-1]["close"])
    if in_position:
        final_equity = cash + position_size * last_close
    else:
        final_equity = cash

    periods = _periods_per_year(timeframe)
    metrics = compute_performance_metrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=trades,
        equity_curve=equity_curve,
        periods_per_year=periods,
    )
    equity_index = df_sorted.index[1:]
    equity_series = pd.Series(equity_curve, index=equity_index)
    return metrics, equity_series, trades


def get_assets_list() -> List[Tuple[str, Optional[str], str]]:
    """
    Return list of (display_name, yf_ticker or None, asset_type) for all assets.

    Returns:
        List of tuples: (display_name, yf_ticker, "crypto"|"forex"|"indices"|"commodities").
    """
    out: List[Tuple[str, Optional[str], str]] = []
    for symbol in CRYPTO_ASSETS:
        out.append((symbol, None, "crypto"))
    for display_symbol, yf_ticker in FOREX:
        out.append((display_symbol, yf_ticker, "forex"))
    for display_symbol, yf_ticker in INDICES:
        out.append((display_symbol, yf_ticker, "indices"))
    for display_symbol, yf_ticker in COMMODITIES:
        out.append((display_symbol, yf_ticker, "commodities"))
    return out


def fetch_ohlcv(
    asset: str,
    yf_ticker: Optional[str],
    asset_type: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV for the given asset and timeframe.

    Crypto uses Binance; EURUSD/GBPUSD/USDJPY use Massive; other traditional
    assets use yfinance.

    Args:
        asset: Display name (e.g. ETHUSDT, EURUSD).
        yf_ticker: yfinance ticker for traditional assets; None for crypto.
        asset_type: "crypto", "forex", "indices", or "commodities".
        timeframe: "15m", "1h", "4h", "1d".

    Returns:
        OHLCV DataFrame.
    """
    if asset_type == "crypto":
        return fetch_binance_ohlcv(
            asset, timeframe, lookback_years=LOOKBACK_CRYPTO_YEARS
        )
    assert yf_ticker is not None
    global _traditional_lookback_logged
    if not _traditional_lookback_logged:
        _log.info(
            "Using lookback period: %s years for traditional assets",
            LOOKBACK_TRADITIONAL_YEARS,
        )
        _traditional_lookback_logged = True
    if asset_type == "forex" and asset in FOREX_MASSIVE_SYMBOLS:
        _log.info("Using Massive data for %s (%s)", asset, timeframe)
        return fetch_massive_forex_ohlcv(
            asset,
            timeframe,
            lookback_years=LOOKBACK_TRADITIONAL_YEARS,
        )
    return fetch_yfinance_ohlcv(
        yf_ticker, timeframe, lookback_years=LOOKBACK_TRADITIONAL_YEARS
    )


RESULT_CSV_COLUMNS = [
    "asset",
    "timeframe",
    "total_trades",
    "win_rate",
    "profit_factor",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
]


def _print_summary_by_asset(df: pd.DataFrame) -> None:
    """
    Print a summary table grouped by asset: Avg PF, Avg Sharpe, Avg Return.

    Args:
        df: Results DataFrame with columns asset, profit_factor, sharpe_ratio, total_return.
    """
    if df.empty:
        print("No data.")
        return
    sub = df.copy()
    sub["profit_factor"] = sub["profit_factor"].clip(upper=MAX_PROFIT_FACTOR)
    agg = sub.groupby("asset").agg({
        "profit_factor": "mean",
        "sharpe_ratio": "mean",
        "total_return": "mean",
    }).round(4)
    agg.columns = ["Avg PF", "Avg Sharpe", "Avg Return"]
    agg = agg.sort_index().reset_index()
    print(agg.to_string(index=False))
    print()


def run_trend_strategy_backtest(
    results_dir: str = "research_results",
    output_csv: str = "trend_strategy_timeframe_results.csv",
    output_chart: str = "trend_strategy_timeframe_equity.png",
) -> pd.DataFrame:
    """
    Backtest the production MA crossover trend strategy across all assets and timeframes.

    Loops over timeframe and asset; fetches OHLCV (Binance for crypto, Massive for
    forex majors, yfinance for indices/commodities); runs backtest; saves CSV and
    ETHUSDT equity chart; prints per-asset summary (mean PF, Sharpe, return).

    Args:
        results_dir: Directory for CSV and charts (relative to project root).
        output_csv: Filename for results CSV.
        output_chart: Filename for ETHUSDT equity by timeframe chart.

    Returns:
        Results DataFrame with columns RESULT_CSV_COLUMNS (asset, timeframe,
        total_trades, win_rate, profit_factor, max_drawdown, sharpe_ratio, total_return).
    """
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_PROJECT_ROOT, os.path.normpath(results_dir))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, output_csv)

    logging.info(f"DEBUG MODE: {DEBUG_MODE}")
    logging.info(f"Assets: {ASSETS}")
    logging.info(f"Timeframes: {TIMEFRAMES}")

    results: List[Dict[str, Any]] = []
    eth_equity: Dict[str, pd.Series] = {}

    assets = get_assets_list()

    for timeframe in TIMEFRAMES:
        for item in assets:
            asset, yf_ticker, asset_type = item
            print(f"  {timeframe} / {asset}")
            try:
                ohlcv = fetch_ohlcv(asset, yf_ticker, asset_type, timeframe)
                if len(ohlcv) < MIN_BARS:
                    raise ValueError(
                        f"Insufficient bars: {len(ohlcv)} (need {MIN_BARS}+)"
                    )
                time.sleep(REQUEST_DELAY)
                metrics, equity_series, _ = run_single_backtest(
                    ohlcv, asset, timeframe
                )
                pf = metrics.get("profit_factor")
                if pf is not None and isinstance(pf, float) and math.isinf(pf):
                    metrics["profit_factor"] = MAX_PROFIT_FACTOR
                row = {
                    "asset": asset,
                    "timeframe": timeframe,
                    "total_trades": int(round(metrics["total_trades"])),
                    "win_rate": metrics["win_rate"],
                    "profit_factor": metrics["profit_factor"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "total_return": metrics["total_return"],
                }
                results.append(row)
                if asset == "ETHUSDT":
                    eth_equity[timeframe] = equity_series
            except Exception as e:
                print(f"    Failed: {e}")

    if not results:
        print("No backtest results to save.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[RESULT_CSV_COLUMNS]
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'.")

    # ETHUSDT equity curves for all timeframes
    if eth_equity:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for tf in TIMEFRAMES:
            if tf not in eth_equity:
                continue
            eq = eth_equity[tf]
            if eq.empty or eq.iloc[0] == 0:
                continue
            norm = eq / eq.iloc[0] * 100.0
            ax.plot(norm.index, norm.values, label=tf, alpha=0.8)
        ax.set_title(
            "Trend strategy (MA crossover) — ETHUSDT equity by timeframe (normalized to 100)"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (index)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        chart_path = os.path.join(results_dir, output_chart)
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Chart saved to '{chart_path}'.")

    # Summary table grouped by asset (helps compare contributors vs laggards)
    print("\n" + "=" * 60)
    print("Trend strategy summary (grouped by asset)")
    print("=" * 60)
    _print_summary_by_asset(df)

    return df


def main() -> None:
    """Entry point: run trend strategy backtest across assets and timeframes."""
    _env = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.isfile(_env):
        load_dotenv(dotenv_path=_env)
    print("Trend strategy (MA crossover) backtest")
    print("Timeframes: 15m, 1h, 4h, 1d")
    print(
        "Assets: crypto (Binance), forex majors (Massive), "
        "indices & commodities (yfinance)\n"
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_trend_strategy_backtest(
        results_dir="research_results",
        output_csv="trend_strategy_timeframe_results.csv",
        output_chart="trend_strategy_timeframe_equity.png",
    )


if __name__ == "__main__":
    main()
