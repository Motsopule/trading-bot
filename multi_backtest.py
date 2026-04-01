"""
Run the MA crossover backtest across multiple trading pairs and summarize results.

This script reuses the existing backtest engine in `backtest.py` (including
`TradingStrategy` and risk-style position sizing). Data sources:
- Crypto: Binance public klines REST API.
- Forex, indices, commodities: yfinance (4h bars via 1h resample).

No live trading or authenticated API is used.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from backtest import BacktestConfig, run_backtest, run_backtest_with_ohlcv
from research.data_providers.http_retry import yfinance_download_with_retries

# Criteria for selecting production markets (Task 3)
MIN_TRADES_FOR_PRODUCTION = 10
MAX_ACCEPTABLE_DRAWDOWN_PERCENT = -40.0  # e.g. -40% max drawdown acceptable
TOP_N_PRODUCTION_MARKETS = 5

# Timeframe for all backtests (4h)
TIMEFRAME = "4h"

# Crypto pairs (Binance)
PAIRS: List[str] = [
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "BTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "ADAUSDT",
]

# Forex: (display symbol for logs, yfinance ticker)
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
        lookback_years=int(os.getenv("BACKTEST_LOOKBACK_YEARS", "8")),
        trading_fee_percent=float(os.getenv("TRADING_FEE_PERCENT", "0.1")),
        slippage_percent=float(os.getenv("SLIPPAGE_PERCENT", "0.05")),
    )

    return base_config


def _daily_to_pseudo_4h(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily OHLCV bars into pseudo 4h bars (6 bars per day).

    Each daily candle is expanded into 6 synthetic 4h bars with the same
    open, high, low, close; volume is split evenly (volume/6). Timestamps
    are spaced every 4 hours within the day (00:00, 04:00, 08:00, 12:00,
    16:00, 20:00 UTC).

    Args:
        daily_df: DataFrame with open, high, low, close, volume and
            DatetimeIndex (date or datetime).

    Returns:
        DataFrame with same columns and 6 rows per original row, index UTC.
    """
    if daily_df.empty:
        return daily_df
    # 4h offsets within a day (UTC): 0, 4, 8, 12, 16, 20
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
    lookback_years: int = 10,
) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance and return 4h bars for strategy compatibility.

    Yahoo Finance only provides 1h data for the last ~730 days. For long
    historical windows we use a fallback:

    - If lookback_years <= 2: download 1h data and resample to 4h (true
      intraday bars).
    - If lookback_years > 2: download 1d data and convert each daily candle
      into 6 pseudo-4h bars (same open/high/low/close per bar, volume/6,
      timestamps every 4h within the day). This allows long backtests to run
      without 1h data availability.

    Args:
        yf_ticker: yfinance symbol (e.g. 'EURUSD=X', '^GSPC', 'GC=F').
        lookback_years: Years of history to request.

    Returns:
        DataFrame with open, high, low, close, volume and datetime index (UTC).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * lookback_years)
    use_daily_fallback = lookback_years > 2

    if use_daily_fallback:
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
            f"Insufficient yfinance data for {yf_ticker}: got "
            f"{len(dl) if dl is not None else 0} bars"
        )
    # Normalize column names (yfinance uses Open, High, Low, Close, Volume)
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
        # Resample 1h to 4h
        if dl.index.inferred_freq != "D" and len(dl) > 4 * 200:
            resampled = dl.resample("4h").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            ).dropna(how="all")
            resampled["volume"] = resampled["volume"].fillna(0)
            dl = resampled.dropna(subset=["open", "high", "low", "close"])

    if dl.index.tz is None:
        dl.index = dl.index.tz_localize("UTC", ambiguous="raise")
    return dl


def _print_results_table(results: List[Dict[str, float]]) -> None:
    """
    Print a formatted summary table to the console.

    Args:
        results: List of per-pair result dictionaries.
    """
    has_sharpe = results and "sharpe_ratio" in results[0]
    header = (
        f"{'Pair':<12}"
        f"{'Trades':>8}"
        f"{'WinRate':>10}"
        f"{'PF':>8}"
        f"{'Drawdown':>10}"
        f"{'Return':>9}"
    )
    if has_sharpe:
        header += f"{'Sharpe':>8}"
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
            f"{row['pair']:<12}"
            f"{int(row['total_trades']):>8}"
            f"{win_rate_str:>10}"
            f"{pf_str:>8}"
            f"{drawdown_str:>10}"
            f"{return_str:>9}"
        )
        if has_sharpe:
            line += f"{row.get('sharpe_ratio', 0):>8.2f}"
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


def select_production_markets(
    results: List[Dict[str, float]],
    min_trades: int = MIN_TRADES_FOR_PRODUCTION,
    max_drawdown_percent: float = MAX_ACCEPTABLE_DRAWDOWN_PERCENT,
    top_n: int = TOP_N_PRODUCTION_MARKETS,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    From backtest results, select top 3-5 assets for production:
    - highest profit factor
    - acceptable drawdown (e.g. max drawdown >= max_drawdown_percent)
    - sufficient trade frequency (min_trades)

    Returns:
        (list of selected pair symbols, list of their result dicts).
    """
    # Filter: sufficient trades and acceptable drawdown
    filtered = [
        r
        for r in results
        if r["total_trades"] >= min_trades
        and r["max_drawdown"] >= max_drawdown_percent
    ]
    # Sort by profit factor descending (inf last or first — put inf first)
    def _pf_sort_key(r: Dict[str, float]) -> Tuple[int, float]:
        pf = r["profit_factor"]
        return (0 if math.isinf(pf) else 1, -pf if not math.isinf(pf) else 0)

    filtered.sort(key=_pf_sort_key)
    # Take top N
    selected = filtered[:top_n]
    symbols = [r["pair"] for r in selected]
    return symbols, selected


def _print_and_save_production_markets(
    results: List[Dict[str, float]],
    output_path: str = "production_markets.txt",
) -> None:
    """Identify top 3-5 production markets and print/save them."""
    symbols, selected = select_production_markets(results)
    if not selected:
        print(
            "\nNo assets met production criteria "
            f"(min_trades>={MIN_TRADES_FOR_PRODUCTION}, "
            f"max_drawdown>={MAX_ACCEPTABLE_DRAWDOWN_PERCENT}%)."
        )
        return
    print("\n--- Best strategy markets (production candidates) ---")
    print(
        f"Criteria: profit factor (high), drawdown>={MAX_ACCEPTABLE_DRAWDOWN_PERCENT}%, "
        f"min_trades>={MIN_TRADES_FOR_PRODUCTION}"
    )
    for r in selected:
        pf_str = "Inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
        print(
            f"  {r['pair']}: PF={pf_str}, trades={int(r['total_trades'])}, "
            f"drawdown={r['max_drawdown']:.1f}%, return={r['total_return']:.1f}%"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Top production markets (TRADING_PAIRS for multi-asset bot)\n")
        f.write(",".join(symbols) + "\n")
    print(f"\nProduction markets list saved to '{output_path}' (use as TRADING_PAIRS).")


def _run_one_crypto(
    pair: str,
    base_config: BacktestConfig,
    lookback_years: Optional[int] = None,
    verbose: bool = True,
    return_equity: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[pd.Series]]:
    """Run backtest for one crypto pair (Binance). Returns (result dict, equity_series or None)."""
    config = BacktestConfig(
        symbol=pair,
        interval=base_config.interval,
        initial_capital=base_config.initial_capital,
        atr_multiplier=base_config.atr_multiplier,
        daily_loss_limit_percent=base_config.daily_loss_limit_percent,
        risk_per_trade_percent=base_config.risk_per_trade_percent,
        max_position_exposure_percent=base_config.max_position_exposure_percent,
        lookback_years=lookback_years or base_config.lookback_years,
        trade_log_csv=f"backtest_trades_{pair}.csv",
        trading_fee_percent=base_config.trading_fee_percent,
        slippage_percent=base_config.slippage_percent,
    )
    try:
        _, metrics, equity_series = run_backtest(config)
        if not verbose:
            pass  # run_backtest always prints; we could add verbose to backtest later
    except Exception as exc:  # noqa: BLE001
        print(f"Backtest failed for {pair}: {exc}")
        return None, None
    res = _metrics_to_result(pair, metrics)
    eq = equity_series if return_equity else None
    return res, eq


def _run_one_yfinance(
    display_symbol: str,
    yf_ticker: str,
    base_config: BacktestConfig,
    lookback_years: int,
    verbose: bool = True,
    return_equity: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[pd.Series]]:
    """Run backtest for one traditional asset (yfinance). Returns (result dict, equity_series or None)."""
    config = BacktestConfig(
        symbol=display_symbol,
        interval=TIMEFRAME,
        initial_capital=base_config.initial_capital,
        atr_multiplier=base_config.atr_multiplier,
        daily_loss_limit_percent=base_config.daily_loss_limit_percent,
        risk_per_trade_percent=base_config.risk_per_trade_percent,
        max_position_exposure_percent=base_config.max_position_exposure_percent,
        lookback_years=lookback_years,
        trade_log_csv=f"backtest_trades_{display_symbol}.csv",
        trading_fee_percent=base_config.trading_fee_percent,
        slippage_percent=base_config.slippage_percent,
    )
    try:
        ohlcv = fetch_yfinance_4h(yf_ticker, lookback_years=lookback_years)
        if verbose:
            print(
                f"Fetched {len(ohlcv)} 4h bars for {display_symbol} ({yf_ticker})"
            )
        _, metrics, equity_series = run_backtest_with_ohlcv(
            ohlcv, config, verbose=verbose
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Backtest failed for {display_symbol} ({yf_ticker}): {exc}")
        return None, None
    res = _metrics_to_result(display_symbol, metrics)
    eq = equity_series if return_equity else None
    return res, eq


def _metrics_to_result(pair: str, metrics: Dict[str, float]) -> Dict[str, float]:
    """Build result row from metrics dict (include sharpe_ratio if present)."""
    row: Dict[str, float] = {
        "pair": pair,
        "total_trades": float(metrics["total_trades"]),
        "win_rate": float(metrics["win_rate"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "total_return": float(metrics["total_return"]),
    }
    if "sharpe_ratio" in metrics:
        row["sharpe_ratio"] = float(metrics["sharpe_ratio"])
    return row


def run_multi_backtest(
    lookback_crypto: Optional[int] = None,
    lookback_traditional: int = 10,
    crypto_only: bool = False,
    include_forex: bool = True,
    include_indices: bool = True,
    include_commodities: bool = True,
) -> List[Dict[str, float]]:
    """
    Run the backtest engine for all configured assets.

    Crypto: Binance klines (4h). Traditional: yfinance 4h (1h resampled).
    Same strategy and risk logic for all. Optional long history: 10y traditional,
    max available for crypto.

    Args:
        lookback_crypto: Years for crypto (default from env or 8).
        lookback_traditional: Years for forex/indices/commodities (default 10).
        crypto_only: If True, run only crypto pairs.
        include_forex: Include forex pairs.
        include_indices: Include index pairs.
        include_commodities: Include commodity pairs.

    Returns:
        List of result dictionaries, one per successfully backtested asset.
    """
    base_config = _load_base_config_from_env()
    if lookback_crypto is None:
        lookback_crypto = base_config.lookback_years

    results: List[Dict[str, float]] = []

    for pair in PAIRS:
        print(f"\n=== Running backtest for {pair} (crypto) ===")
        res, _ = _run_one_crypto(pair, base_config, lookback_years=lookback_crypto)
        if res is not None:
            results.append(res)

    if not crypto_only:
        if include_forex:
            for display_symbol, yf_ticker in FOREX:
                print(f"\n=== Running backtest for {display_symbol} (forex) ===")
                res, _ = _run_one_yfinance(
                    display_symbol, yf_ticker, base_config, lookback_traditional
                )
                if res is not None:
                    results.append(res)
        if include_indices:
            for display_symbol, yf_ticker in INDICES:
                print(f"\n=== Running backtest for {display_symbol} (index) ===")
                res, _ = _run_one_yfinance(
                    display_symbol, yf_ticker, base_config, lookback_traditional
                )
                if res is not None:
                    results.append(res)
        if include_commodities:
            for display_symbol, yf_ticker in COMMODITIES:
                print(f"\n=== Running backtest for {display_symbol} (commodity) ===")
                res, _ = _run_one_yfinance(
                    display_symbol, yf_ticker, base_config, lookback_traditional
                )
                if res is not None:
                    results.append(res)

    if not results:
        print("No successful backtests were completed.")
        return results

    print("\nMulti-asset backtest summary:\n")
    _print_results_table(results)

    csv_path = "multi_asset_results.csv"
    _save_results_csv(results, csv_path)
    print(f"\nSummary results saved to '{csv_path}'.")

    # Production selection only over crypto pairs
    crypto_results = [r for r in results if r["pair"] in PAIRS]
    if crypto_results:
        _print_and_save_production_markets(crypto_results)

    return results


def run_multi_backtest_with_equity(
    lookback_crypto: Optional[int] = None,
    lookback_traditional: int = 10,
    crypto_only: bool = False,
    include_forex: bool = True,
    include_indices: bool = True,
    include_commodities: bool = True,
    verbose: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, pd.Series]]:
    """
    Run backtests for all configured assets and return results plus equity curves.

    Same as run_multi_backtest but returns a dict of asset -> equity Series
    for plotting (equity curve per asset, drawdown, portfolio).

    Returns:
        (results list, equity_curves dict keyed by asset symbol).
    """
    base_config = _load_base_config_from_env()
    if lookback_crypto is None:
        lookback_crypto = base_config.lookback_years

    results: List[Dict[str, float]] = []
    equity_curves: Dict[str, pd.Series] = {}

    for pair in PAIRS:
        if verbose:
            print(f"\n=== Running backtest for {pair} (crypto) ===")
        res, eq = _run_one_crypto(
            pair, base_config, lookback_years=lookback_crypto,
            verbose=verbose, return_equity=True,
        )
        if res is not None:
            results.append(res)
            if eq is not None:
                equity_curves[pair] = eq

    if not crypto_only:
        if include_forex:
            for display_symbol, yf_ticker in FOREX:
                if verbose:
                    print(f"\n=== Running backtest for {display_symbol} (forex) ===")
                res, eq = _run_one_yfinance(
                    display_symbol, yf_ticker, base_config, lookback_traditional,
                    verbose=verbose, return_equity=True,
                )
                if res is not None:
                    results.append(res)
                    if eq is not None:
                        equity_curves[display_symbol] = eq
        if include_indices:
            for display_symbol, yf_ticker in INDICES:
                if verbose:
                    print(f"\n=== Running backtest for {display_symbol} (index) ===")
                res, eq = _run_one_yfinance(
                    display_symbol, yf_ticker, base_config, lookback_traditional,
                    verbose=verbose, return_equity=True,
                )
                if res is not None:
                    results.append(res)
                    if eq is not None:
                        equity_curves[display_symbol] = eq
        if include_commodities:
            for display_symbol, yf_ticker in COMMODITIES:
                if verbose:
                    print(f"\n=== Running backtest for {display_symbol} (commodity) ===")
                res, eq = _run_one_yfinance(
                    display_symbol, yf_ticker, base_config, lookback_traditional,
                    verbose=verbose, return_equity=True,
                )
                if res is not None:
                    results.append(res)
                    if eq is not None:
                        equity_curves[display_symbol] = eq

    return results, equity_curves


if __name__ == "__main__":
    run_multi_backtest(
        lookback_crypto=8,
        lookback_traditional=10,
    )

