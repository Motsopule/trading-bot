"""
Data module for fetching and processing market data from Binance API.

This module handles all interactions with the Binance API for retrieving
candlestick data, account information, and current prices.
"""

import logging
import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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

        logger.info(f"DataFetcher initialized (testnet={testnet})")

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

    def _require_symbol_timeframe(self) -> bool:
        """Return True if symbol and timeframe are set; log and return False otherwise."""
        if not self.symbol or not self.timeframe:
            logger.warning(
                "API call skipped: symbol and timeframe must be set "
                f"(symbol={self.symbol!r}, timeframe={self.timeframe!r})"
            )
            return False
        return True

    def get_klines(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch candlestick data from Binance with retries.

        Retries up to 3 times on failure with a short delay between attempts.
        Fails gracefully after retries exhausted.

        Args:
            limit: Number of klines to fetch (default: 500)

        Returns:
            DataFrame with OHLCV data, or None if error occurs after retries
        """
        if not self._require_symbol_timeframe():
            return None

        max_retries = 3
        retry_delay_seconds = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    limit=limit
                )

                df = pd.DataFrame(
                    klines,
                    columns=[
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ]
                )

                # Convert to numeric types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Drop rows with NaN in OHLCV to avoid bad indicator inputs
                df = df.dropna(subset=numeric_columns)

                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')

                # Keep only necessary columns
                df = df[['open', 'high', 'low', 'close', 'volume']]

                self.consecutive_errors = 0
                logger.info(
                    "symbol=%s timeframe=%s candles_fetched=%d",
                    self.symbol, self.timeframe, len(df)
                )
                return df

            except BinanceAPIException as e:
                self.consecutive_errors += 1
                logger.warning(
                    "symbol=%s timeframe=%s get_klines retry attempt=%d/%d "
                    "BinanceAPIException: %s",
                    self.symbol, self.timeframe, attempt, max_retries, e
                )
                if attempt < max_retries:
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error(
                        "symbol=%s timeframe=%s get_klines failed after %d "
                        "retries (last error: %s)",
                        self.symbol, self.timeframe, max_retries, e
                    )
                    return None

            except Exception as e:
                self.consecutive_errors += 1
                logger.warning(
                    "symbol=%s timeframe=%s get_klines retry attempt=%d/%d "
                    "error: %s",
                    self.symbol, self.timeframe, attempt, max_retries, e
                )
                if attempt < max_retries:
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error(
                        "symbol=%s timeframe=%s get_klines failed after %d "
                        "retries (last error: %s)",
                        self.symbol, self.timeframe, max_retries, e
                    )
                    return None

        return None

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
