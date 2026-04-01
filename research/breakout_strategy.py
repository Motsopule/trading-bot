"""
Volatility breakout research strategy (Donchian Channel + ATR filter).

This module is for research only. It is not used by the production trading bot.
Timeframe: 4H.

Indicators:
- Donchian 20 High/Low (shift(1) to avoid lookahead)
- Donchian 10 Exit High/Low (shift(1))
- ATR(20), ATR_MA = ATR(20).rolling(10).mean()

Entry: Long when close > previous_20_high and ATR(20) > ATR_MA.
       Short when close < previous_20_low and ATR(20) > ATR_MA.
Exit:  Long when close < previous_10_low; Short when close > previous_10_high.
Stop:  2 × ATR(20) from entry (fixed at entry).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# Donchian and ATR periods
DONCHIAN_ENTRY_PERIOD = 20
DONCHIAN_EXIT_PERIOD = 10
ATR_PERIOD = 20
ATR_MA_PERIOD = 10

# Stop loss: 2 × ATR(20)
ATR_STOP_MULTIPLIER = 2.0

# Position sizing: 1% risk per trade
RISK_PER_TRADE_PERCENT = 1.0


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Donchian and ATR indicators to the OHLCV DataFrame.

    All Donchian calculations use shift(1) to prevent lookahead bias.
    Expects columns: open, high, low, close, volume.
    Adds: previous_20_high, previous_20_low, previous_10_high, previous_10_low,
          atr_20, atr_ma.

    Args:
        df: DataFrame with open, high, low, close, volume.

    Returns:
        DataFrame with original columns plus indicator columns.
    """
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]

    # Donchian 20 High/Low (shift(1) = prior bars only)
    out["previous_20_high"] = high.shift(1).rolling(DONCHIAN_ENTRY_PERIOD).max()
    out["previous_20_low"] = low.shift(1).rolling(DONCHIAN_ENTRY_PERIOD).min()

    # Donchian 10 Exit High/Low (shift(1))
    out["previous_10_high"] = high.shift(1).rolling(DONCHIAN_EXIT_PERIOD).max()
    out["previous_10_low"] = low.shift(1).rolling(DONCHIAN_EXIT_PERIOD).min()

    # ATR(20)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_20"] = tr.rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # ATR Moving Average
    out["atr_ma"] = out["atr_20"].rolling(ATR_MA_PERIOD).mean()

    return out


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate long/short entry and exit signals from indicator DataFrame.

    Long entry:  close > previous_20_high AND ATR(20) > ATR_MA.
    Short entry: close < previous_20_low AND ATR(20) > ATR_MA.
    Long exit:   close < previous_10_low.
    Short exit:  close > previous_10_high.

    Args:
        df: DataFrame with close, previous_20_high, previous_20_low,
            previous_10_high, previous_10_low, atr_20, atr_ma.

    Returns:
        DataFrame with added columns: long_entry, short_entry, long_exit,
        short_exit (bool).
    """
    out = df.copy()
    close = out["close"]
    prev_20_high = out["previous_20_high"]
    prev_20_low = out["previous_20_low"]
    prev_10_high = out["previous_10_high"]
    prev_10_low = out["previous_10_low"]
    atr = out["atr_20"]
    atr_ma = out["atr_ma"]

    volatility_filter = atr > atr_ma

    out["long_entry"] = (close > prev_20_high) & volatility_filter
    out["short_entry"] = (close < prev_20_low) & volatility_filter
    out["long_exit"] = close < prev_10_low
    out["short_exit"] = close > prev_10_high

    return out


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    side: Literal["long", "short"] = "long",
) -> float:
    """
    Compute stop-loss price fixed at entry (2 × ATR(20) from entry).

    long_stop = entry_price - 2 * atr
    short_stop = entry_price + 2 * atr

    Args:
        entry_price: Entry price of the trade.
        atr: ATR(20) at the entry candle.
        side: "long" or "short".

    Returns:
        Stop-loss price level.
    """
    if side == "long":
        return entry_price - ATR_STOP_MULTIPLIER * atr
    return entry_price + ATR_STOP_MULTIPLIER * atr
