"""
Data module for fetching and processing market data from Binance API.

This module handles all interactions with the Binance API for retrieving
candlestick data, account information, and current prices.
"""

import json
import logging
import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def log_event(event: dict) -> None:
    """Structured JSON log line for observability; never raises."""
    try:
        logger.info(json.dumps(event))
    except Exception:
        logger.info(str(event))


def validate_time_continuity(df: pd.DataFrame, interval: str) -> None:
    """Ensure candle open times advance by exactly one interval with no gaps."""
    timeframe_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    expected_freq = timeframe_map.get(interval)

    if expected_freq is None:
        return

    diffs = df.index.to_series().diff().dropna()

    expected_delta = pd.Timedelta(expected_freq)

    if not (diffs == expected_delta).all():
        raise RuntimeError("CRITICAL: time gap detected in candles")


class DataFetcher:
    """
    Handles data fetching from Binance API.
    
    Attributes:
        client: Binance API client instance
        symbol: Trading pair symbol (e.g., 'ETHUSDT')
        timeframe: Kline interval (e.g., '4h')
        consecutive_errors: Counter for consecutive API errors
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the DataFetcher with Binance API credentials.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use Binance testnet (default: True)
        """
        self.testnet = testnet
        if testnet:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True
            )
            # Force Spot Testnet URL (python-binance can default to live despite testnet=True)
            self.client.API_URL = "https://testnet.binance.vision/api"
        else:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret
            )

        self.symbol = None
        self.timeframe = None
        self.consecutive_errors = 0
        # Minimum position value (USDT) to count as a real position; avoids dust
        self.min_position_value_usdt = 10.0

        logger.info(
            "DataFetcher initialized testnet=%s API_URL=%s",
            testnet,
            getattr(self.client, "API_URL", "unknown"),
        )

    def set_trading_pair(self, symbol: str, timeframe: str):
        """
        Set the trading pair and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            timeframe: Kline interval (e.g., '4h')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        logger.info(f"Trading pair set: {symbol} on {timeframe}")

    def get_candles(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Exchange-agnostic: fetch candle data for a symbol.

        Sets the internal symbol/timeframe and returns OHLCV candles.
        Allows the same interface for Binance, Forex, Deriv, or other backends.

        Args:
            symbol: Trading pair symbol (e.g. 'ETHUSDT')
            timeframe: Optional interval (e.g. '4h'); uses instance default if None
            limit: Number of candles (default 500)

        Returns:
            DataFrame with OHLCV data, or None if fetch fails
        """
        tf = timeframe or self.timeframe or "4h"
        self.set_trading_pair(symbol, tf)
        return self.get_klines(limit=limit)

    def _require_symbol_timeframe(self) -> bool:
        """Return True if symbol and timeframe are set; log and return False otherwise."""
        if not self.symbol or not self.timeframe:
            logger.warning(
                "API call skipped: symbol and timeframe must be set "
                f"(symbol={self.symbol!r}, timeframe={self.timeframe!r})"
            )
            return False
        return True

    def safe_get_klines(self, params: Dict, retries: int = 3) -> List:
        """
        Call Binance get_klines with exponential backoff; re-raises on final failure.
        """
        for attempt in range(retries):
            try:
                return self.client.get_klines(**params)
            except Exception:
                if attempt == retries - 1:
                    raise

                sleep_time = 2 ** attempt
                time.sleep(sleep_time)

    def get_klines_full(self, symbol: str, interval: str, required: int = 500) -> List:
        """
        Fetch historical klines using pagination to guarantee required candle count.
        """
        if not symbol or not interval:
            raise ValueError("symbol and interval must be set")

        all_klines: List = []
        end_time: Optional[int] = None

        max_loops = 10  # safety to prevent infinite loops

        for _ in range(max_loops):
            params: Dict = {
                "symbol": symbol,
                "interval": interval,
                "limit": 1000,
            }

            if end_time is not None:
                params["endTime"] = end_time

            batch = self.safe_get_klines(params)

            if batch is None:
                raise RuntimeError("CRITICAL: API returned None")

            if not batch:
                break

            all_klines = batch + all_klines

            # Move backwards in time
            end_time = batch[0][0] - 1

            if len(all_klines) >= required:
                break

            # If batch smaller than max, no more data
            if len(batch) < 1000:
                break

        if len(all_klines) < required:
            raise RuntimeError(
                f"CRITICAL: insufficient klines fetched ({len(all_klines)} < {required})"
            )

        log_event(
            {
                "event": "data_fetch",
                "symbol": symbol,
                "interval": interval,
                "candles_fetched": len(all_klines),
            }
        )

        return all_klines[-required:]

    def klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Convert raw Binance kline rows to a typed DataFrame (open_time as datetime)."""
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)

        return df

    def _klines_to_ohlcv_indexed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy trading DataFrame: DatetimeIndex named 'datetime', OHLCV only.
        Raises if OHLCV contains NaN (no silent row drops).
        """
        numeric_cols = ["open", "high", "low", "close", "volume"]
        if df[numeric_cols].isna().any().any():
            raise RuntimeError(
                "CRITICAL: invalid OHLCV data (NaN in open/high/low/close/volume)"
            )

        out = df.copy()
        out["datetime"] = out["open_time"]
        out = out.set_index("datetime")
        out = out[["open", "high", "low", "close", "volume"]]
        return out

    def get_klines(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch candlestick data using paginated history; guarantees `limit` closed rows.

        Returns:
            DataFrame with OHLCV indexed by datetime, or None if symbol/timeframe unset.
        Raises:
            RuntimeError if history is insufficient, data is invalid, or API fails after retries.
        """
        if not self._require_symbol_timeframe():
            return None

        assert self.symbol is not None and self.timeframe is not None

        try:
            start_time = time.time()
            klines = self.get_klines_full(
                self.symbol, self.timeframe, required=limit
            )
            df = self.klines_to_dataframe(klines)
            df = self._klines_to_ohlcv_indexed(df)

            if df.index.duplicated().any():
                raise RuntimeError("CRITICAL: duplicate candles detected")

            validate_time_continuity(df, self.timeframe)

            # HARD VALIDATION
            if len(df) < 200:
                raise RuntimeError(
                    f"CRITICAL: insufficient candles for indicators ({len(df)})"
                )

            duration = time.time() - start_time

            self.consecutive_errors = 0
            logger.info(
                "symbol=%s timeframe=%s candles_fetched=%d",
                self.symbol,
                self.timeframe,
                len(df),
            )
            log_event(
                {
                    "event": "data_fetch_complete",
                    "symbol": self.symbol,
                    "interval": self.timeframe,
                    "candles_returned": len(df),
                    "duration_sec": round(duration, 3),
                }
            )
            return df

        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(
                "symbol=%s timeframe=%s get_klines BinanceAPIException: %s",
                self.symbol,
                self.timeframe,
                e,
            )
            raise RuntimeError(f"CRITICAL: Binance klines fetch failed: {e}") from e

        except RuntimeError:
            self.consecutive_errors += 1
            raise

        except Exception as e:
            self.consecutive_errors += 1
            logger.error(
                "symbol=%s timeframe=%s get_klines error: %s",
                self.symbol,
                self.timeframe,
                e,
            )
            raise RuntimeError(f"CRITICAL: klines fetch failed: {e}") from e

    def get_current_price(self) -> Optional[float]:
        """
        Get the current price of the trading pair.
        
        Returns:
            Current price as float, or None if error occurs
        """
        if not self._require_symbol_timeframe():
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            
            self.consecutive_errors = 0
            logger.debug(
                "symbol=%s timeframe=%s current_price=%.8f",
                self.symbol, self.timeframe, price
            )
            return price

        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(
                "symbol=%s timeframe=%s price_fetch_failed BinanceAPIException: %s",
                self.symbol, self.timeframe, e
            )
            return None

        except Exception as e:
            self.consecutive_errors += 1
            logger.error(
                "symbol=%s timeframe=%s price_fetch_failed error: %s",
                self.symbol, self.timeframe, e
            )
            return None

    def get_account_balance(self) -> Optional[Dict[str, float]]:
        """
        Get account balances for USDT and the base asset.
        
        Returns:
            Dictionary with 'USDT' and base asset balances, or None if error
        """
        if not self._require_symbol_timeframe():
            return None
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                if asset == 'USDT' or asset == self.symbol.replace('USDT', ''):
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    balances[asset] = free + locked
            
            self.consecutive_errors = 0
            logger.debug(f"Account balances: {balances}")
            
            return balances
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error fetching account: {e}")
            return None
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error fetching account: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders for the trading pair.
        
        Returns:
            List of open orders, or empty list if error occurs
        """
        if not self._require_symbol_timeframe():
            return []
        try:
            orders = self.client.get_open_orders(symbol=self.symbol)
            self.consecutive_errors = 0
            logger.debug(f"Open orders: {len(orders)}")
            return orders
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error fetching open orders: {e}")
            return []
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error fetching open orders: {e}")
            return []

    def get_open_positions(self) -> Optional[Dict]:
        """
        Get current position information for the trading pair.
        
        Returns:
            Dictionary with position info, or None if error occurs
        """
        try:
            # For spot trading, we check account balance
            balances = self.get_account_balance()
            if balances is None:
                return None
            
            base_asset = self.symbol.replace('USDT', '')
            base_balance = balances.get(base_asset, 0.0)
            usdt_balance = balances.get('USDT', 0.0)
            
            current_price = self.get_current_price()
            if current_price is None:
                return None
            
            position_value = base_balance * current_price
            has_position = (
                base_balance > 0.0
                and position_value >= self.min_position_value_usdt
            )

            position = {
                'base_balance': base_balance,
                'usdt_balance': usdt_balance,
                'position_value': position_value,
                'current_price': current_price,
                'has_position': has_position
            }
            
            logger.debug(f"Position info: {position}")
            return position
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None

    def reset_error_count(self):
        """Reset the consecutive error counter."""
        self.consecutive_errors = 0
        logger.debug("Error count reset")
