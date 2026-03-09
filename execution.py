"""
Execution module for placing and managing orders on Binance.

This module handles order placement, cancellation, and position management.
"""

import logging
import time
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles order execution on Binance.
    Supports real execution and paper trading (simulated fills at current price).
    
    Attributes:
        client: Binance API client instance
        symbol: Trading pair symbol
        paper_trading: If True, orders are simulated; no orders sent to Binance
        consecutive_errors: Counter for consecutive API errors
    """
    
    def __init__(self, client: Client, symbol: str, paper_trading: bool = False):
        """
        Initialize the OrderExecutor and cache symbol trading info once.
        
        Args:
            client: Binance API client instance
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            paper_trading: If True, simulate fills at current market price; do not send orders
        """
        self.client = client
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.consecutive_errors = 0
        self._symbol_info: Optional[Dict] = None
        self._paper_order_id = 0
        self._load_symbol_info()
        mode = "paper" if paper_trading else "live"
        logger.info(f"OrderExecutor initialized for {symbol} (mode={mode})")

    def _load_symbol_info(self) -> None:
        """Fetch and cache symbol trading info from exchange (once)."""
        try:
            exchange_info = self.client.get_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == self.symbol:
                    self._symbol_info = s
                    logger.debug(f"Cached symbol info for {self.symbol}")
                    return
            logger.warning(f"Symbol {self.symbol} not found in exchange info")
        except Exception as e:
            logger.error(f"Error loading symbol info: {e}")

    def get_symbol_info(self) -> Optional[Dict]:
        """
        Get cached symbol trading information (precision, min quantity, etc.).
        
        Returns:
            Dictionary with symbol info, or None if not available
        """
        return self._symbol_info

    def get_min_notional(self) -> Optional[float]:
        """
        Get minimum notional (quote asset) for the symbol from NOTIONAL or MIN_NOTIONAL filter.
        
        Returns:
            Minimum notional in quote asset (e.g. USDT), or None if not found
        """
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return None
        for f in symbol_info.get('filters', []):
            if f.get('filterType') in ('NOTIONAL', 'MIN_NOTIONAL'):
                min_notional = f.get('minNotional') or f.get('notional')
                if min_notional is not None:
                    return float(min_notional)
        return None

    def round_quantity(self, quantity: float) -> float:
        """
        Round quantity to symbol's step size.
        
        Args:
            quantity: Raw quantity to round
            
        Returns:
            Rounded quantity
        """
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return round(quantity, 6)  # Default precision
        
        try:
            # Get step size from filters
            for filter_item in symbol_info['filters']:
                if filter_item['filterType'] == 'LOT_SIZE':
                    step_size = float(filter_item['stepSize'])
                    # Round to step size
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    return round(quantity - (quantity % step_size), precision)
            
            return round(quantity, 6)
            
        except Exception as e:
            logger.warning(f"Error rounding quantity: {e}")
            return round(quantity, 6)

    def round_price(self, price: float) -> float:
        """
        Round price to symbol's tick size.
        
        Args:
            price: Raw price to round
            
        Returns:
            Rounded price
        """
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return round(price, 2)  # Default precision
        
        try:
            # Get tick size from filters
            for filter_item in symbol_info['filters']:
                if filter_item['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter_item['tickSize'])
                    # Round to tick size
                    precision = len(str(tick_size).split('.')[-1].rstrip('0'))
                    return round(price - (price % tick_size), precision)
            
            return round(price, 2)
            
        except Exception as e:
            logger.warning(f"Error rounding price: {e}")
            return round(price, 2)

    def _get_simulation_price(self) -> Optional[float]:
        """
        Get current market price for paper trading fills.
        Uses ticker price only; deterministic, no randomness.
        
        Returns:
            Current price from ticker, or None if fetch fails
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            price = float(ticker.get('price', 0))
            if price <= 0:
                return None
            return self.round_price(price)
        except Exception as e:
            logger.error(f"Error getting simulation price: {e}")
            return None

    def _make_paper_order_response(
        self,
        side: str,
        quantity: float,
        fill_price: float
    ) -> Dict:
        """
        Build a realistic order response dict for paper trades.
        Same shape as live order response for compatibility.
        """
        self._paper_order_id += 1
        return {
            'order_id': self._paper_order_id,
            'symbol': self.symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'price': fill_price,
            'status': 'FILLED',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _weighted_fill_price(self, order: Dict) -> float:
        """
        Compute volume-weighted average fill price from order fills.
        Falls back to cummulativeQuoteQty/executedQty if fills missing.
        """
        fills = order.get('fills') or []
        if fills:
            total_qty = 0.0
            weighted_sum = 0.0
            for f in fills:
                qty = float(f.get('qty', 0))
                price = float(f.get('price', 0))
                weighted_sum += price * qty
                total_qty += qty
            if total_qty > 0:
                return weighted_sum / total_qty
        # Binance market orders: use quote quantity / executed quantity (key is cummulativeQuoteQty)
        exec_qty = float(order.get('executedQty', 0) or 0)
        quote_qty = float(order.get('cummulativeQuoteQty', 0) or 0)
        if exec_qty > 0 and quote_qty > 0:
            return quote_qty / exec_qty
        return float(order.get('price', 0) or 0)

    def _check_min_notional(self, quantity: float, price: float) -> Tuple[bool, Optional[str]]:
        """Return (True, None) if notional is valid; (False, reason) otherwise."""
        min_notional = self.get_min_notional()
        if min_notional is None:
            return True, None
        notional = quantity * price
        if notional < min_notional:
            return False, f"Notional {notional:.2f} below minimum {min_notional:.2f}"
        return True, None

    def validate_order_quantity(
        self,
        quantity: float,
        price: float,
        side: str
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Validate quantity and notional before placing an order.
        Applies rounding first, then checks quantity > 0 and notional >= minimum.

        Args:
            quantity: Raw quantity in base asset
            price: Current or approximate price for notional check
            side: 'BUY' or 'SELL' for logging

        Returns:
            (valid, rounded_quantity, reason_if_invalid)
        """
        if quantity is None or not isinstance(quantity, (int, float)):
            return False, None, "Quantity must be a number"
        if quantity <= 0:
            return False, None, f"Quantity must be > 0 (got {quantity})"
        rounded = self.round_quantity(float(quantity))
        if rounded <= 0:
            return False, None, (
                f"Quantity rounded to zero or negative (raw={quantity}, "
                f"rounded={rounded})"
            )
        ok, reason = self._check_min_notional(rounded, price)
        if not ok:
            return False, None, reason
        return True, rounded, None

    def place_market_buy_order(self, quantity: float) -> Optional[Dict]:
        """
        Place a market buy order.
        In paper mode: simulate fill at current ticker price; do not send to Binance.
        In live mode: send order to Binance and return real fill.

        Validates quantity > 0, rounded quantity, and notional >= minimum before sending.

        Args:
            quantity: Quantity to buy (in base asset)

        Returns:
            Order result dictionary with price and quantity, or None if error
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            approx_price = float(ticker.get('price', 0))
            valid, rounded_qty, reason = self.validate_order_quantity(
                quantity, approx_price, 'BUY'
            )
            if not valid:
                logger.warning(
                    "symbol=%s order=BUY validation_failed: %s",
                    self.symbol, reason
                )
                return None
            quantity = rounded_qty

            if self.paper_trading:
                fill_price = self._get_simulation_price()
                if fill_price is None:
                    logger.error("Paper buy: could not get simulation price")
                    return None
                self.consecutive_errors = 0
                response = self._make_paper_order_response('BUY', quantity, fill_price)
                logger.info(
                    "symbol=%s state=PAPER_ENTRY order=BUY simulated_fill: "
                    "qty=%.6f price=%.4f",
                    self.symbol, quantity, fill_price
                )
                return response

            order = self.client.order_market_buy(
                symbol=self.symbol,
                quantity=quantity
            )

            fill_price = self._weighted_fill_price(order)
            executed_qty = float(order.get('executedQty', 0) or 0)
            
            self.consecutive_errors = 0
            logger.info(f"Market buy order placed: {executed_qty} {self.symbol} @ ~{fill_price:.4f}")
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': executed_qty,
                'price': fill_price,
                'status': order['status'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error placing buy order: {e}")
            return None
            
        except BinanceOrderException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance order error placing buy order: {e}")
            return None
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error placing buy order: {e}")
            return None

    def place_market_sell_order(self, quantity: float) -> Optional[Dict]:
        """
        Place a market sell order.
        In paper mode: simulate fill at current ticker price; do not send to Binance.
        In live mode: send order to Binance and return real fill.

        Validates quantity > 0, rounded quantity, and notional >= minimum before sending.

        Args:
            quantity: Quantity to sell (in base asset)

        Returns:
            Order result dictionary with price and quantity, or None if error
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            approx_price = float(ticker.get('price', 0))
            valid, rounded_qty, reason = self.validate_order_quantity(
                quantity, approx_price, 'SELL'
            )
            if not valid:
                logger.warning(
                    "symbol=%s order=SELL validation_failed: %s",
                    self.symbol, reason
                )
                return None
            quantity = rounded_qty

            if self.paper_trading:
                fill_price = self._get_simulation_price()
                if fill_price is None:
                    logger.error("Paper sell: could not get simulation price")
                    return None
                self.consecutive_errors = 0
                response = self._make_paper_order_response('SELL', quantity, fill_price)
                logger.info(
                    "symbol=%s state=PAPER_EXIT order=SELL simulated_fill: "
                    "qty=%.6f price=%.4f",
                    self.symbol, quantity, fill_price
                )
                return response

            order = self.client.order_market_sell(
                symbol=self.symbol,
                quantity=quantity
            )

            fill_price = self._weighted_fill_price(order)
            executed_qty = float(order.get('executedQty', 0) or 0)
            
            self.consecutive_errors = 0
            logger.info(f"Market sell order placed: {executed_qty} {self.symbol} @ ~{fill_price:.4f}")
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': executed_qty,
                'price': fill_price,
                'status': order['status'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error placing sell order: {e}")
            return None
            
        except BinanceOrderException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance order error placing sell order: {e}")
            return None
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error placing sell order: {e}")
            return None

    def place_stop_loss_order(
        self,
        quantity: float,
        stop_price: float
    ) -> Optional[Dict]:
        """
        Note: Binance spot trading doesn't support stop-market orders directly.
        This method returns stop-loss information that will be monitored in main.py.
        The actual stop-loss execution is handled by monitoring price in the main loop.
        
        Args:
            quantity: Quantity to sell (in base asset)
            stop_price: Stop price trigger
            
        Returns:
            Dictionary with stop-loss information for monitoring
        """
        try:
            quantity = self.round_quantity(quantity)
            stop_price = self.round_price(stop_price)
            
            if quantity <= 0 or stop_price <= 0:
                logger.warning(
                    f"Invalid parameters for stop-loss: qty={quantity}, price={stop_price}"
                )
                return None
            
            # Return stop-loss info for monitoring (not an actual order)
            logger.info(
                f"Stop-loss set for monitoring: {quantity} @ {stop_price} {self.symbol}"
            )
            
            return {
                'symbol': self.symbol,
                'side': 'SELL',
                'type': 'STOP_LOSS_MONITOR',
                'quantity': quantity,
                'stop_price': stop_price,
                'status': 'MONITORING',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error setting stop-loss: {e}")
            return None

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an open order.
        In paper mode: no-op (no orders on exchange); returns True.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if self.paper_trading:
            logger.debug(f"Paper mode: cancel_order no-op for {order_id}")
            return True
        try:
            result = self.client.cancel_order(
                symbol=self.symbol,
                orderId=order_id
            )
            
            self.consecutive_errors = 0
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error cancelling order: {e}")
            return False
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error cancelling order: {e}")
            return False

    def cancel_all_open_orders(self) -> bool:
        """
        Cancel all open orders for the symbol.
        In paper mode: no-op; returns True.
        
        Returns:
            True if successful, False otherwise
        """
        if self.paper_trading:
            logger.debug("Paper mode: cancel_all_open_orders no-op")
            return True
        try:
            result = self.client.cancel_open_orders(symbol=self.symbol)
            self.consecutive_errors = 0
            logger.info("All open orders cancelled")
            return True
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error cancelling all orders: {e}")
            return False
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error cancelling all orders: {e}")
            return False

    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """
        Get status of an order.
        In paper mode: returns None (no real orders on exchange).
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order information dictionary, or None if error
        """
        if self.paper_trading:
            return None
        try:
            order = self.client.get_order(symbol=self.symbol, orderId=order_id)
            
            self.consecutive_errors = 0
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'quantity': float(order['origQty']),
                'executed_quantity': float(order['executedQty']),
                'price': float(order.get('price', 0)),
                'status': order['status'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except BinanceAPIException as e:
            self.consecutive_errors += 1
            logger.error(f"Binance API error getting order status: {e}")
            return None
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Unexpected error getting order status: {e}")
            return None

    def reset_error_count(self):
        """Reset the consecutive error counter."""
        self.consecutive_errors = 0
        logger.debug("Error count reset")
