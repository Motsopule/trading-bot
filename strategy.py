"""
Strategy module implementing the MA crossover trading strategy.

This module calculates moving averages, ATR, and generates entry/exit signals
based on the specified strategy rules.
"""

import logging
from typing import Optional, Dict, Tuple, Any
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator

logger = logging.getLogger(__name__)


def daily_trend_bullish(symbol: str, data_client: Any) -> bool:
    """
    Higher timeframe confirmation: True when Daily MA50 > Daily MA200.
    Used to allow only LONG trades when daily trend is bullish.
    """
    df_daily = data_client.get_candles(symbol, timeframe="1d", limit=200)
    if df_daily is None or len(df_daily) < 200:
        return False
    df_daily = df_daily.copy()
    df_daily["ma50"] = df_daily["close"].rolling(50).mean()
    df_daily["ma200"] = df_daily["close"].rolling(200).mean()
    return bool(df_daily["ma50"].iloc[-1] > df_daily["ma200"].iloc[-1])


class TradingStrategy:
    """
    Implements the MA crossover trading strategy (long and short).

    Long Strategy Rules:
    - Entry: 50MA > 200MA, 20MA crosses above 50MA, ATR(14) > ATR_MA(20)
    - Exit:  20MA crosses below 50MA OR close < 50MA OR stop-loss (2x ATR)

    Short Strategy Rules:
    - Entry: 50MA < 200MA, 20MA crosses below 50MA, ATR(14) > ATR_MA(20)
    - Exit:  20MA crosses above 50MA OR close > 50MA OR stop-loss (2x ATR)
    """

    ATR_MA_PERIOD = 20

    def __init__(self, atr_multiplier: float = 2.0):
        """
        Initialize the trading strategy.

        Args:
            atr_multiplier: Multiplier for ATR stop-loss calculation (default: 2.0)
        """
        self.atr_multiplier = atr_multiplier
        self.ma_20_period = 20
        self.ma_50_period = 50
        self.ma_200_period = 200
        self.atr_period = 14
        
        logger.info(f"TradingStrategy initialized with ATR multiplier: {atr_multiplier}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for the strategy.

        Does not run if insufficient data for longest MA (200). Validates
        ATR and MAs on the last bar to prevent NaN propagation into signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns (unchanged if insufficient data)
        """
        if df is None or len(df) < self.ma_200_period:
            logger.warning(
                "Insufficient data for indicator calculation: need %d bars, got %s",
                self.ma_200_period, len(df) if df is not None else "None"
            )
            return df

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy()

        try:
            # Calculate Moving Averages
            sma_20 = SMAIndicator(close=df['close'], window=self.ma_20_period)
            sma_50 = SMAIndicator(close=df['close'], window=self.ma_50_period)
            sma_200 = SMAIndicator(close=df['close'], window=self.ma_200_period)

            df['ma_20'] = sma_20.sma_indicator()
            df['ma_50'] = sma_50.sma_indicator()
            df['ma_200'] = sma_200.sma_indicator()

            # Calculate ATR
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.atr_period
            )
            df['atr'] = atr.average_true_range()
            # ATR(14) 20-period moving average for volatility expansion filter
            df['atr_ma'] = df['atr'].rolling(self.ATR_MA_PERIOD).mean()

            # Calculate RSI(14) using only past (closed) candles
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))

            # Prevent NaN propagation: ensure last bar has valid indicators
            last = df.iloc[-1]
            indicator_cols = ['ma_20', 'ma_50', 'ma_200', 'atr', 'atr_ma', 'rsi']
            for col in indicator_cols:
                val = last.get(col)
                if pd.isna(val) or (col == 'atr' and (val is None or val <= 0)) or (
                    col == 'atr_ma' and (val is None or val <= 0 or not np.isfinite(val))
                ):
                    logger.warning(
                        "Indicator safety: last bar has invalid %s (value=%s); "
                        "signals will be suppressed",
                        col, val
                    )
                    break

            logger.debug("Indicators calculated successfully")
            return df

        except Exception as e:
            logger.error("Error calculating indicators: %s", e)
            return df

    def score_signal(self, df: pd.DataFrame) -> int:
        """
        Score a signal for ranking (higher = stronger).

        Rules:
        - Trend strength: MA50 > MA200 → +3
        - Distance from MA50: (close - MA50)/MA50 > 0.02 → +2
        - ATR expansion: ATR > rolling ATR mean(20) → +2
        - Momentum: current close > previous close → +1

        Returns:
            Total score (0–8).
        """
        if df is None or len(df) < 2:
            return 0
        current = df.iloc[-1]
        previous = df.iloc[-2]
        score = 0
        try:
            ma_50 = current.get("ma_50")
            ma_200 = current.get("ma_200")
            if ma_50 is not None and ma_200 is not None and not pd.isna(ma_50) and not pd.isna(ma_200):
                if ma_50 > ma_200:
                    score += 3
            close = current.get("close")
            if close is not None and ma_50 is not None and not pd.isna(close) and not pd.isna(ma_50) and ma_50 > 0:
                if (close - ma_50) / ma_50 > 0.02:
                    score += 2
            atr = current.get("atr")
            atr_ma = current.get("atr_ma")
            if atr is not None and atr_ma is not None and not pd.isna(atr) and not pd.isna(atr_ma):
                if atr > atr_ma:
                    score += 2
            prev_close = previous.get("close")
            if close is not None and prev_close is not None and not pd.isna(close) and not pd.isna(prev_close):
                if close > prev_close:
                    score += 1
        except (TypeError, ZeroDivisionError):
            pass
        return score

    def check_entry_signal(
        self,
        df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        data_client: Optional[Any] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if entry signal conditions are met.
        
        Long entry conditions:
        1. Trend filter: 50MA above 200MA on the current closed candle
        2. Momentum: 20MA crosses above 50MA
           (previous 20MA <= previous 50MA and current 20MA > current 50MA)
        3. Volatility expansion: ATR(14) > ATR_MA(20) on current closed candle
        4. Daily trend filter (when symbol/data_client provided): Daily MA50 > Daily MA200

        Args:
            df: Current DataFrame with indicators
            previous_df: Previous DataFrame for crossover detection
            symbol: Optional symbol for daily trend check (long only)
            data_client: Optional data client for fetching daily klines
            
        Returns:
            Tuple of (signal_present, signal_details)
        """
        if df is None or len(df) < 2:
            return False, None
        
        try:
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current
            
            # ATR and MAs must be valid before use (prevents NaN propagation)
            if (
                pd.isna(current.get('ma_200'))
                or pd.isna(current.get('ma_20'))
                or pd.isna(current.get('ma_50'))
                or pd.isna(previous.get('ma_20'))
                or pd.isna(previous.get('ma_50'))
            ):
                return False, None
            atr = current.get('atr')
            if atr is None or pd.isna(atr) or not np.isfinite(atr) or atr <= 0:
                return False, None
            atr_ma = current.get('atr_ma')
            if atr_ma is None or pd.isna(atr_ma) or not np.isfinite(atr_ma) or atr_ma <= 0:
                return False, None

            rsi_val = current.get('rsi')
            if rsi_val is None or pd.isna(rsi_val) or not np.isfinite(rsi_val):
                return False, None

            # Trend filter: 50MA above 200MA on current closed candle
            trend_filter_ok = current['ma_50'] > current['ma_200']

            # Volatility expansion: ATR(14) > ATR_MA(20) on current closed candle
            volatility_filter_ok = current['atr'] > current['atr_ma']

            # Momentum filter: RSI(14) must be above 50 on current closed candle
            rsi_momentum_ok = rsi_val > 50

            # Momentum: 20MA crosses above 50MA using previous vs current values
            ma_bullish_cross = (
                previous['ma_20'] <= previous['ma_50']
                and current['ma_20'] > current['ma_50']
            )

            # Breakout confirmation: close above recent 20-candle high (exclude current candle)
            recent_high = df["high"].rolling(20).max().iloc[-2] if len(df) >= 21 else None
            breakout = (
                recent_high is not None
                and not pd.isna(recent_high)
                and current["close"] > recent_high
            )

            if trend_filter_ok and volatility_filter_ok and ma_bullish_cross and rsi_momentum_ok and breakout:
                # Daily trend filter for long: only allow when Daily MA50 > Daily MA200
                if symbol and data_client:
                    if not daily_trend_bullish(symbol, data_client):
                        logger.debug(
                            "Daily trend check: symbol=%s trend=bearish — signal rejected",
                            symbol,
                        )
                        return False, None
                    logger.debug(
                        "Daily trend check: symbol=%s trend=bullish",
                        symbol,
                    )
                signal_details = {
                    'entry_price': current['close'],
                    'ma_20': current['ma_20'],
                    'ma_50': current['ma_50'],
                    'ma_200': current['ma_200'],
                    'atr': current['atr'],
                    'rsi': rsi_val,
                    'side': 'long',
                    'timestamp': current.name
                }
                
                logger.info("Entry signal detected")
                return True, signal_details
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking entry signal: {e}")
            return False, None

    def check_short_entry_signal(
        self,
        df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if short entry signal conditions are met.

        Short entry conditions:
        1. Trend filter: 50MA below 200MA on the current closed candle
        2. Momentum: 20MA crosses below 50MA
           (previous 20MA >= previous 50MA and current 20MA < current 50MA)
        3. Volatility expansion: ATR(14) > ATR_MA(20) on current closed candle

        Args:
            df: Current DataFrame with indicators
            previous_df: Previous DataFrame for crossover detection

        Returns:
            Tuple of (signal_present, signal_details)
        """
        if df is None or len(df) < 2:
            return False, None

        try:
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current

            # ATR and MAs must be valid before use (prevents NaN propagation)
            if (
                pd.isna(current.get('ma_200'))
                or pd.isna(current.get('ma_20'))
                or pd.isna(current.get('ma_50'))
                or pd.isna(previous.get('ma_20'))
                or pd.isna(previous.get('ma_50'))
            ):
                return False, None
            atr = current.get('atr')
            if atr is None or pd.isna(atr) or not np.isfinite(atr) or atr <= 0:
                return False, None
            atr_ma = current.get('atr_ma')
            if atr_ma is None or pd.isna(atr_ma) or not np.isfinite(atr_ma) or atr_ma <= 0:
                return False, None

            # Trend filter: 50MA below 200MA on current closed candle
            trend_filter_ok = current['ma_50'] < current['ma_200']

            # Volatility expansion: ATR(14) > ATR_MA(20) on current closed candle
            volatility_filter_ok = current['atr'] > current['atr_ma']

            # Momentum: 20MA crosses below 50MA using previous vs current values
            ma_bearish_cross = (
                previous['ma_20'] >= previous['ma_50']
                and current['ma_20'] < current['ma_50']
            )

            if trend_filter_ok and volatility_filter_ok and ma_bearish_cross:
                signal_details = {
                    'entry_price': current['close'],
                    'ma_20': current['ma_20'],
                    'ma_50': current['ma_50'],
                    'ma_200': current['ma_200'],
                    'atr': current['atr'],
                    'side': 'short',
                    'timestamp': current.name
                }

                logger.info("Short entry signal detected")
                return True, signal_details

            return False, None

        except Exception as e:
            logger.error(f"Error checking short entry signal: {e}")
            return False, None

    def check_exit_signal(
        self,
        df: pd.DataFrame,
        entry_price: float
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if long exit signal conditions are met.

        Long exit conditions:
        1. 20MA crosses below 50MA
           (previous 20MA >= previous 50MA and current 20MA < current 50MA)
        2. Close price drops below 50MA on the current closed candle
        
        Args:
            df: Current DataFrame with indicators
            entry_price: Price at which position was entered
            
        Returns:
            Tuple of (signal_present, signal_details)
        """
        if df is None or len(df) < 2:
            return False, None
        
        try:
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current
            
            # MAs must be valid before use (prevents NaN propagation)
            for key in ('ma_20', 'ma_50'):
                curr_val = current.get(key)
                prev_val = previous.get(key)
                if (
                    curr_val is None or pd.isna(curr_val) or not np.isfinite(curr_val)
                    or prev_val is None or pd.isna(prev_val) or not np.isfinite(prev_val)
                ):
                    return False, None
            
            # Condition 1: 20MA crosses below 50MA
            ma_bearish_cross = (
                previous['ma_20'] >= previous['ma_50'] and
                current['ma_20'] < current['ma_50']
            )

            # Condition 2: Close drops below 50MA on current closed candle
            trailing_trend_exit = current['close'] < current['ma_50']

            if ma_bearish_cross or trailing_trend_exit:
                signal_details = {
                    'exit_price': current['close'],
                    'ma_20': current['ma_20'],
                    'ma_50': current['ma_50'],
                    'entry_price': entry_price,
                    'side': 'long',
                    'timestamp': current.name
                }
                
                logger.info("Exit signal detected")
                return True, signal_details
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return False, None

    def check_short_exit_signal(
        self,
        df: pd.DataFrame,
        entry_price: float
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if short exit signal conditions are met.

        Short exit conditions:
        1. 20MA crosses above 50MA
           (previous 20MA <= previous 50MA and current 20MA > current 50MA)
        2. Close price rises above 50MA on the current closed candle

        Args:
            df: Current DataFrame with indicators
            entry_price: Price at which position was entered

        Returns:
            Tuple of (signal_present, signal_details)
        """
        if df is None or len(df) < 2:
            return False, None

        try:
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current

            # MAs must be valid before use (prevents NaN propagation)
            for key in ('ma_20', 'ma_50'):
                curr_val = current.get(key)
                prev_val = previous.get(key)
                if (
                    curr_val is None or pd.isna(curr_val) or not np.isfinite(curr_val)
                    or prev_val is None or pd.isna(prev_val) or not np.isfinite(prev_val)
                ):
                    return False, None

            # Condition 1: 20MA crosses above 50MA
            ma_bullish_cross = (
                previous['ma_20'] <= previous['ma_50']
                and current['ma_20'] > current['ma_50']
            )

            # Condition 2: Close rises above 50MA on current closed candle
            trailing_trend_exit = current['close'] > current['ma_50']

            if ma_bullish_cross or trailing_trend_exit:
                signal_details = {
                    'exit_price': current['close'],
                    'ma_20': current['ma_20'],
                    'ma_50': current['ma_50'],
                    'entry_price': entry_price,
                    'side': 'short',
                    'timestamp': current.name
                }

                logger.info("Short exit signal detected")
                return True, signal_details

            return False, None

        except Exception as e:
            logger.error(f"Error checking short exit signal: {e}")
            return False, None

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str = "long") -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            entry_price: Entry price for the position
            atr: Current ATR value
            side: Position side ("long" or "short")
            
        Returns:
            Stop-loss price
        """
        if atr is None or pd.isna(atr) or not np.isfinite(atr) or atr <= 0:
            logger.warning("Invalid ATR value, using 2% stop-loss")
            if side == "short":
                return entry_price * 1.02
            return entry_price * 0.98

        if side == "short":
            stop_loss = entry_price + (self.atr_multiplier * atr)
        else:
            stop_loss = entry_price - (self.atr_multiplier * atr)

        logger.debug(
            "Stop-loss calculated: %s (entry: %s, ATR: %s, side: %s)",
            stop_loss,
            entry_price,
            atr,
            side,
        )

        return max(0, stop_loss)  # Ensure stop-loss is not negative

    def check_stop_loss(self, current_price: float, stop_loss: float, side: str = "long") -> bool:
        """
        Check if stop-loss has been hit.
        
        Args:
            current_price: Current market price
            stop_loss: Stop-loss price
            side: Position side ("long" or "short")
            
        Returns:
            True if stop-loss is hit, False otherwise
        """
        if side == "short":
            if current_price >= stop_loss:
                logger.info("Short stop-loss hit: %s >= %s", current_price, stop_loss)
                return True
        else:
            if current_price <= stop_loss:
                logger.info("Stop-loss hit: %s <= %s", current_price, stop_loss)
                return True
        return False

    def get_position_status(
        self,
        df: pd.DataFrame,
        entry_price: float,
        stop_loss: float
    ) -> Dict:
        """
        Get current status of an open position.
        
        Args:
            df: Current DataFrame with indicators
            entry_price: Entry price of the position
            stop_loss: Stop-loss price
            
        Returns:
            Dictionary with position status information
        """
        if df is None or len(df) == 0:
            return {}
        
        try:
            current = df.iloc[-1]
            current_price = current['close']
            
            # Calculate P&L
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            
            # Check if stop-loss is hit
            stop_loss_hit = self.check_stop_loss(current_price, stop_loss)
            
            # Check exit signal
            exit_signal, exit_details = self.check_exit_signal(df, entry_price)
            
            status = {
                'current_price': current_price,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'pnl_percent': pnl_percent,
                'stop_loss_hit': stop_loss_hit,
                'exit_signal': exit_signal,
                'exit_details': exit_details,
                'timestamp': current.name
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting position status: {e}")
            return {}
