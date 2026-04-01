"""
Mean reversion research strategy (RSI + MA50 + ATR).

This module is for research only. It is not used by the production trading bot.
Indicators: RSI(14), MA50, ATR(14).
Entry/exit rules and fixed ATR-based stop loss at entry.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# Parameterized thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ATR_STOP_MULT = 1.5

# Indicator periods
RSI_PERIOD = 14
MA_PERIOD = 50
ATR_PERIOD = 14

# Position sizing: 1% risk per trade
RISK_PER_TRADE_PERCENT = 1.0


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI(14), MA50, and ATR(14) to the OHLCV DataFrame.

    Expects columns: open, high, low, close, volume.
    Adds columns: rsi_14, ma50, atr_14.

    Args:
        df: DataFrame with open, high, low, close, volume.

    Returns:
        DataFrame with original columns plus rsi_14, ma50, atr_14.
    """
    out = df.copy()
    close = out["close"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi_14"] = out["rsi_14"].fillna(50.0)

    # MA50
    out["ma50"] = close.rolling(window=MA_PERIOD, min_periods=MA_PERIOD).mean()

    # ATR(14)
    high = out["high"]
    low = out["low"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    return out


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate long/short entry and exit signals from indicator DataFrame.

    Long entry:  close < ma50 - atr_14  and  RSI < RSI_OVERSOLD (30).
    Short entry: close > ma50 + atr_14  and  RSI > RSI_OVERBOUGHT (70).
    Long exit:   close >= ma50.
    Short exit: close <= ma50.

    Args:
        df: DataFrame with close, ma50, atr_14, rsi_14 (e.g. from calculate_indicators).

    Returns:
        DataFrame with added columns: long_entry, short_entry, long_exit, short_exit (bool).
    """
    out = df.copy()
    close = out["close"]
    ma50 = out["ma50"]
    atr = out["atr_14"]
    rsi = out["rsi_14"]

    out["long_entry"] = (
        (close < ma50 - atr) & (rsi < RSI_OVERSOLD)
    )
    out["short_entry"] = (
        (close > ma50 + atr) & (rsi > RSI_OVERBOUGHT)
    )
    out["long_exit"] = close >= ma50
    out["short_exit"] = close <= ma50

    return out


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    side: Literal["long", "short"] = "long",
) -> float:
    """
    Compute stop-loss price fixed at entry (1.5 × ATR from entry).

    For long: stop = entry_price - ATR_STOP_MULT * atr.
    For short: stop = entry_price + ATR_STOP_MULT * atr.

    Args:
        entry_price: Entry price of the trade.
        atr: ATR(14) at the entry candle (used for fixed stop).
        side: "long" or "short".

    Returns:
        Stop-loss price level.
    """
    if side == "long":
        return entry_price - ATR_STOP_MULT * atr
    return entry_price + ATR_STOP_MULT * atr
