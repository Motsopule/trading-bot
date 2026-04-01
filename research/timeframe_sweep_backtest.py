"""
Timeframe sweep backtest for breakout and mean reversion research strategies.

Runs both strategies across multiple timeframes (15m, 1h, 4h, 1d) and all
configured assets. Saves results to CSV, generates ETHUSDT equity charts
per strategy, and prints summary tables. Research-only; no production files
are modified.

No lookahead bias: Donchian uses shift(1); mean reversion uses rolling only.
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
from research.breakout_backtest import (
    run_single_backtest as run_breakout_backtest,
)
from research.mean_reversion_backtest import (
    run_single_backtest as run_mean_reversion_backtest,
)

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

# Cap profit_factor for CSV (avoid inf)
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
        return 365.0 * 24 * (60 // 15)  # 96 per day
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

    Each bar gets the same OHLC as the day; volume is split equally.
    Used for traditional assets when the requested timeframe is intraday
    but only daily data is available (long lookback).

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

    Uses daily data with pseudo intraday bars when needed (long lookback).
    No lookahead: pseudo bars repeat the same OHLC per day.

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
    # For long lookback, yfinance only has reliable daily data
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
        # Short lookback: try 1h for 1h/4h
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
        # Resample 1h to 4h if needed
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


def recompute_sharpe(equity_series: pd.Series, timeframe: str) -> float:
    """
    Compute annualized Sharpe ratio from equity series using correct bar frequency.

    Args:
        equity_series: Equity curve indexed by time.
        timeframe: One of "15m", "1h", "4h", "1d".

    Returns:
        Sharpe ratio (0 if insufficient data or zero variance).
    """
    if equity_series is None or len(equity_series) < 2 or equity_series.iloc[0] <= 0:
        return 0.0
    bar_returns = equity_series.pct_change().dropna()
    if len(bar_returns) == 0 or bar_returns.std() == 0:
        return 0.0
    periods = _periods_per_year(timeframe)
    return float(bar_returns.mean() / bar_returns.std() * math.sqrt(periods))


def run_one_backtest(
    strategy: str,
    ohlcv: pd.DataFrame,
    asset: str,
    timeframe: str,
) -> Tuple[Dict[str, float], Optional[pd.Series]]:
    """
    Run a single backtest for the given strategy, OHLCV, asset and timeframe.

    Recomputes Sharpe ratio using the correct periods per year for the timeframe.

    Args:
        strategy: "breakout" or "mean_reversion".
        ohlcv: OHLCV DataFrame (open, high, low, close, volume).
        asset: Display name of the asset.
        timeframe: "15m", "1h", "4h", or "1d".

    Returns:
        (metrics_dict with total_trades, win_rate, profit_factor, max_drawdown,
         sharpe_ratio, total_return; equity_series for charting).
    """
    if strategy == "breakout":
        metrics, equity_series, _ = run_breakout_backtest(ohlcv, asset)
    elif strategy == "mean_reversion":
        metrics, equity_series, _ = run_mean_reversion_backtest(ohlcv, asset)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    metrics["sharpe_ratio"] = recompute_sharpe(equity_series, timeframe)
    pf = metrics.get("profit_factor")
    if pf is not None and (isinstance(pf, float) and math.isinf(pf)):
        metrics["profit_factor"] = MAX_PROFIT_FACTOR
    return metrics, equity_series


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
    assets use yfinance with pseudo bars when intraday and lookback is long.

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


def run_timeframe_sweep(
    results_dir: str = "research_results",
    output_csv: str = "strategy_timeframe_comparison.csv",
    breakout_chart: str = "breakout_timeframe_equity.png",
    mean_reversion_chart: str = "mean_reversion_timeframe_equity.png",
) -> pd.DataFrame:
    """
    Run full timeframe sweep for both strategies and all assets.

    Loops over strategy -> timeframe -> asset; fetches OHLCV; runs backtest;
    saves CSV; generates ETHUSDT equity charts per strategy; prints summary tables.

    Args:
        results_dir: Directory for CSV and charts (relative to project root).
        output_csv: Filename for results CSV.
        breakout_chart: Filename for breakout ETHUSDT equity chart.
        mean_reversion_chart: Filename for mean reversion ETHUSDT equity chart.

    Returns:
        Results DataFrame.
    """
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_PROJECT_ROOT, os.path.normpath(results_dir))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, output_csv)

    logging.info(f"DEBUG MODE: {DEBUG_MODE}")
    logging.info(f"Assets: {ASSETS}")
    logging.info(f"Timeframes: {TIMEFRAMES}")

    results: List[Dict[str, Any]] = []
    # ETHUSDT equity per (strategy, timeframe) for charts
    eth_equity: Dict[str, Dict[str, pd.Series]] = {
        "breakout": {},
        "mean_reversion": {},
    }

    assets = get_assets_list()

    for strategy in ["breakout", "mean_reversion"]:
        for timeframe in TIMEFRAMES:
            for item in assets:
                asset, yf_ticker, asset_type = item
                print(f"  {strategy} / {timeframe} / {asset}")
                try:
                    ohlcv = fetch_ohlcv(asset, yf_ticker, asset_type, timeframe)
                    if len(ohlcv) < MIN_BARS:
                        raise ValueError(
                            f"Insufficient bars: {len(ohlcv)} (need {MIN_BARS}+)"
                        )
                    time.sleep(REQUEST_DELAY)
                    metrics, equity_series = run_one_backtest(
                        strategy, ohlcv, asset, timeframe
                    )
                    row = {
                        "strategy": strategy,
                        "asset": asset,
                        "timeframe": timeframe,
                        "total_trades": metrics["total_trades"],
                        "win_rate": metrics["win_rate"],
                        "profit_factor": metrics["profit_factor"],
                        "max_drawdown": metrics["max_drawdown"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "total_return": metrics["total_return"],
                    }
                    results.append(row)
                    if asset == "ETHUSDT" and equity_series is not None:
                        eth_equity[strategy][timeframe] = equity_series
                except Exception as e:
                    print(f"    Failed: {e}")

    if not results:
        print("No backtest results to save.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'.")

    # ETHUSDT equity charts: one per strategy, all timeframes
    for strategy, chart_name in [
        ("breakout", breakout_chart),
        ("mean_reversion", mean_reversion_chart),
    ]:
        curves = eth_equity.get(strategy, {})
        if not curves:
            continue
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for tf in TIMEFRAMES:
            if tf not in curves:
                continue
            eq = curves[tf]
            if eq.empty or eq.iloc[0] == 0:
                continue
            norm = eq / eq.iloc[0] * 100.0
            ax.plot(norm.index, norm.values, label=tf, alpha=0.8)
        title = (
            "Breakout strategy — ETHUSDT equity by timeframe (normalized to 100)"
            if strategy == "breakout"
            else "Mean reversion strategy — ETHUSDT equity by timeframe (normalized to 100)"
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (index)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        chart_path = os.path.join(results_dir, chart_name)
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Chart saved to '{chart_path}'.")

    # Summary tables: by strategy and timeframe (Avg PF, Avg Sharpe, Avg Return)
    print("\n" + "=" * 60)
    print("Breakout Strategy Summary")
    print("=" * 60)
    _print_summary_table(df[df["strategy"] == "breakout"])

    print("\n" + "=" * 60)
    print("Mean Reversion Strategy Summary")
    print("=" * 60)
    _print_summary_table(df[df["strategy"] == "mean_reversion"])

    return df


def _print_summary_table(sub: pd.DataFrame) -> None:
    """
    Print a summary table: Timeframe | Avg PF | Avg Sharpe | Avg Return.

    Args:
        sub: DataFrame filtered to one strategy.
    """
    if sub.empty:
        print("No data.")
        return
    # Cap profit_factor for display
    sub = sub.copy()
    sub["profit_factor"] = sub["profit_factor"].clip(upper=MAX_PROFIT_FACTOR)
    agg = sub.groupby("timeframe").agg({
        "profit_factor": "mean",
        "sharpe_ratio": "mean",
        "total_return": "mean",
    }).round(4)
    agg.columns = ["Avg PF", "Avg Sharpe", "Avg Return"]
    # Reorder to TIMEFRAMES
    agg = agg.reindex([t for t in TIMEFRAMES if t in agg.index])
    print(agg.to_string())
    print()


def main() -> None:
    """Entry point: run timeframe sweep and save results."""
    _env = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.isfile(_env):
        load_dotenv(dotenv_path=_env)
    print("Timeframe sweep: breakout & mean reversion across 15m, 1h, 4h, 1d")
    print(
        "Assets: crypto (Binance), forex majors (Massive), "
        "indices & commodities (yfinance)\n"
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_timeframe_sweep(
        results_dir="research_results",
        output_csv="strategy_timeframe_comparison.csv",
        breakout_chart="breakout_timeframe_equity.png",
        mean_reversion_chart="mean_reversion_timeframe_equity.png",
    )


if __name__ == "__main__":
    main()
