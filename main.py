"""
Main trading bot module that orchestrates data fetching, strategy,
risk management, and order execution.

This is the entry point for the crypto trading bot.
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict
from dotenv import load_dotenv

from data import DataFetcher
from strategy import TradingStrategy
from risk import RiskManager
from execution import OrderExecutor
from journal import write_trade_entry
from notifications import TelegramNotifier

# Load environment variables from project directory (reliable regardless of cwd)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path)
# Optional: debug Telegram env (mask token)
_tok = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if _tok or _chat:
    _masked = f"{_tok[:4]}...{_tok[-4:]}" if len(_tok) > 8 else "(set)"
    logging.getLogger(__name__).debug(
        "Telegram env: TELEGRAM_BOT_TOKEN=%s TELEGRAM_CHAT_ID=%s",
        _masked, _chat or "(empty)"
    )

# Configure logging
def setup_logging():
    """Configure logging to file and console."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(
        log_dir,
        f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    # Structured format: timestamp, level, symbol/timeframe/state in message
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file


class TradingBot:
    """
    Main trading bot class that coordinates all components.
    """
    
    def __init__(self):
        """Initialize the trading bot with all components."""
        # Setup logging
        log_file = setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trading bot starting. Log file: {log_file}")
        
        # Load configuration from environment
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        testnet = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'
        
        trading_pair = os.getenv('TRADING_PAIR', 'ETHUSDT')
        timeframe = os.getenv('TIMEFRAME', '4h')
        paper_trading = os.getenv('PAPER_TRADING', 'False').lower() == 'true'
        initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000.0'))
        
        daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT_PERCENT', '3.0'))
        atr_multiplier = float(os.getenv('STOP_LOSS_ATR_MULTIPLIER', '2.0'))
        
        trading_start_hour = int(os.getenv('TRADING_START_HOUR', '8'))
        trading_end_hour = int(os.getenv('TRADING_END_HOUR', '17'))
        
        max_consecutive_errors = int(os.getenv('MAX_CONSECUTIVE_API_ERRORS', '5'))
        max_price_move = float(os.getenv('MAX_PRICE_MOVE_PERCENT', '10.0'))
        max_consecutive_none_market_data = int(
            os.getenv('MAX_CONSECUTIVE_NONE_MARKET_DATA', '3')
        )
        max_consecutive_price_failures = int(
            os.getenv('MAX_CONSECUTIVE_PRICE_FAILURES', '3')
        )

        # Validate API credentials
        if not api_key or not api_secret:
            self.logger.error("API credentials not found in environment variables")
            sys.exit(1)
        
        # Initialize components
        self.data_fetcher = DataFetcher(api_key, api_secret, testnet)
        self.data_fetcher.set_trading_pair(trading_pair, timeframe)
        
        self.strategy = TradingStrategy(atr_multiplier=atr_multiplier)
        
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            daily_loss_limit_percent=daily_loss_limit,
            trading_start_hour=trading_start_hour,
            trading_end_hour=trading_end_hour,
            max_consecutive_errors=max_consecutive_errors,
            max_price_move_percent=max_price_move,
            max_consecutive_none_market_data=max_consecutive_none_market_data,
            max_consecutive_price_failures=max_consecutive_price_failures
        )
        
        self.executor = OrderExecutor(
            self.data_fetcher.client,
            trading_pair,
            paper_trading=paper_trading
        )
        self.paper_trading = paper_trading

        self.notifier = TelegramNotifier()

        # Bot state
        self.running = False
        self.current_position = None
        self.last_price = None
        self.price_history = []
        self.consecutive_none_market_data = 0
        self.consecutive_none_price = 0
        self.last_status_notify_time = 0.0

        self.logger.info("Trading bot initialized successfully")
        self.logger.info(f"Trading pair: {trading_pair}, Timeframe: {timeframe}")
        self.logger.info(f"Testnet mode: {testnet}")
        self.logger.info(
            "Execution mode: %s",
            "PAPER (simulated)" if paper_trading else "LIVE (real orders)"
        )

    def update_market_data(self) -> bool:
        """
        Fetch and update market data.

        Validates candle count against required MA period; skips indicator
        calculation if insufficient data.

        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = self.data_fetcher.symbol
            timeframe = self.data_fetcher.timeframe
            required_ma_period = self.strategy.ma_200_period

            df = self.data_fetcher.get_klines(limit=500)
            if df is None:
                self.consecutive_none_market_data += 1
                self.logger.warning(
                    "symbol=%s timeframe=%s state=HALT reason=market_data_none "
                    "consecutive_none_klines=%d",
                    symbol, timeframe, self.consecutive_none_market_data
                )
                return False
            self.consecutive_none_market_data = 0

            num_candles = len(df)
            if num_candles < required_ma_period:
                self.logger.warning(
                    "symbol=%s timeframe=%s candles=%d required_ma_period=%d "
                    "reason=insufficient_candles_for_ma",
                    symbol, timeframe, num_candles, required_ma_period
                )
                return False

            # Remove live/incomplete candle so signals are evaluated only on closed candles
            if len(df) > 1:
                df = df.iloc[:-1]
            num_candles = len(df)
            if num_candles < required_ma_period:
                self.logger.warning(
                    "symbol=%s timeframe=%s candles=%d required_ma_period=%d "
                    "reason=insufficient_candles_after_drop",
                    symbol, timeframe, num_candles, required_ma_period
                )
                return False

            # Calculate indicators only when we have sufficient data
            df = self.strategy.calculate_indicators(df)
            
            # Get current price
            current_price = self.data_fetcher.get_current_price()
            if current_price is None:
                self.consecutive_none_price += 1
                self.logger.warning(
                    "symbol=%s timeframe=%s state=HALT reason=price_fetch_failed "
                    "consecutive_none_price=%d",
                    symbol, timeframe, self.consecutive_none_price
                )
                return False
            self.consecutive_none_price = 0
            
            # Track price history for volatility check
            if self.last_price is not None:
                price_change = ((current_price - self.last_price) / self.last_price) * 100
                self.price_history.append(price_change)
                if len(self.price_history) > 10:
                    self.price_history.pop(0)
                
                # Check for large price moves
                if abs(price_change) >= self.risk_manager.max_price_move_percent:
                    self.risk_manager.check_kill_switch(
                        self.data_fetcher.consecutive_errors,
                        price_change_percent=price_change,
                        consecutive_none_market_data=self.consecutive_none_market_data,
                        consecutive_price_failures=self.consecutive_none_price
                    )
            
            self.last_price = current_price
            self.market_data = df
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False

    def check_and_enter_position(self):
        """Check for entry signals and enter position if conditions are met."""
        if self.current_position is not None:
            return  # Already in a position
        
        # Check if we can open a new trade
        can_trade, reason = self.risk_manager.can_open_new_trade()
        if not can_trade:
            self.logger.debug(f"Cannot open new trade: {reason}")
            return
        
        # Check kill switch
        kill_switch, kill_reason = self.risk_manager.check_kill_switch(
            self.data_fetcher.consecutive_errors,
            consecutive_none_market_data=self.consecutive_none_market_data,
            consecutive_price_failures=self.consecutive_none_price
        )
        if kill_switch:
            self.logger.warning(
                "symbol=%s timeframe=%s state=HALT kill_switch_check_failed: %s",
                self.data_fetcher.symbol, self.data_fetcher.timeframe, kill_reason
            )
            return
        
        # Check entry signal
        entry_signal, signal_details = self.strategy.check_entry_signal(
            self.market_data
        )
        
        if not entry_signal:
            return
        
        try:
            # Calculate position size
            stop_loss = self.strategy.calculate_stop_loss(
                signal_details['entry_price'],
                signal_details['atr']
            )
            
            position_size = self.risk_manager.calculate_position_size(
                signal_details['entry_price'],
                stop_loss,
                risk_percent=1.0
            )
            
            if position_size <= 0:
                self.logger.warning("Invalid position size calculated")
                return
            
            # Get account balance (paper: use risk manager capital; live: exchange balance)
            if self.paper_trading:
                usdt_balance = self.risk_manager.get_risk_status()['current_capital']
            else:
                balances = self.data_fetcher.get_account_balance()
                if balances is None:
                    return
                usdt_balance = balances.get('USDT', 0.0)
            required_usdt = position_size * signal_details['entry_price']
            
            if required_usdt > usdt_balance:
                self.logger.warning(
                    f"Insufficient balance: need {required_usdt:.2f}, "
                    f"have {usdt_balance:.2f}"
                )
                return
            
            # Place buy order
            buy_order = self.executor.place_market_buy_order(position_size)
            if buy_order is None:
                self.logger.error("Failed to place buy order")
                return

            # Use filled execution price and executed quantity from order (not ticker)
            entry_price = buy_order.get('price') or signal_details['entry_price']
            executed_qty = buy_order.get('quantity', 0) or 0
            if executed_qty <= 0:
                self.logger.error("Buy order returned no executed quantity")
                return

            # Record position with fill price and executed size
            self.current_position = {
                'entry_price': entry_price,
                'entry_time': datetime.now(timezone.utc),
                'quantity': executed_qty,
                'stop_loss': stop_loss,
                'entry_signal': signal_details,
                'buy_order': buy_order
            }
            
            # Place stop-loss order (use executed quantity for monitoring)
            stop_order = self.executor.place_stop_loss_order(
                executed_qty,
                stop_loss
            )
            
            if stop_order:
                self.current_position['stop_order'] = stop_order
            
            entry_state = "PAPER_ENTRY" if self.paper_trading else "LIVE_ENTRY"
            self.logger.info(
                "symbol=%s timeframe=%s state=%s executed_qty=%.6f "
                "entry_price=%.2f stop_loss=%.2f",
                self.data_fetcher.symbol, self.data_fetcher.timeframe,
                entry_state, executed_qty, entry_price, stop_loss
            )

            try:
                self.notifier.notify_entry(
                    {**self.current_position, 'symbol': self.data_fetcher.symbol}
                )
            except Exception as notify_err:
                self.logger.warning(
                    "state=NOTIFY_FAIL action=notify_entry error=%s", notify_err
                )

        except Exception as e:
            entry_state = "PAPER_ENTRY" if self.paper_trading else "LIVE_ENTRY"
            self.logger.error(
                "symbol=%s timeframe=%s state=%s error: %s",
                self.data_fetcher.symbol, self.data_fetcher.timeframe,
                entry_state, e
            )

    def check_and_exit_position(self):
        """Check for exit signals and exit position if conditions are met."""
        if self.current_position is None:
            return  # No position to exit
        
        try:
            entry_price = self.current_position['entry_price']
            
            # Check exit signal
            exit_signal, exit_details = self.strategy.check_exit_signal(
                self.market_data,
                entry_price
            )
            
            # Check stop-loss
            current_price = self.data_fetcher.get_current_price()
            if current_price is None:
                return
            
            stop_loss_hit = self.strategy.check_stop_loss(
                current_price,
                self.current_position['stop_loss']
            )
            
            # Exit if signal or stop-loss hit
            if exit_signal or stop_loss_hit:
                quantity = self.current_position['quantity']

                # Place sell order
                sell_order = self.executor.place_market_sell_order(quantity)
                if sell_order is None:
                    self.logger.error("Failed to place sell order")
                    return

                # Use filled execution price and executed quantity from sell order
                exit_price = sell_order.get('price') or current_price
                executed_sell_qty = sell_order.get('quantity', 0) or quantity
                pnl = (exit_price - entry_price) * executed_sell_qty
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                
                # Record trade (use executed quantities and fill prices)
                trade_result = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': executed_sell_qty,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'entry_time': self.current_position['entry_time'].isoformat(),
                    'exit_time': datetime.now(timezone.utc).isoformat(),
                    'exit_reason': 'stop_loss' if stop_loss_hit else 'signal',
                    'sell_order': sell_order
                }
                
                self.risk_manager.record_trade(trade_result)

                execution_mode = "PAPER" if self.paper_trading else "LIVE"
                write_trade_entry(
                    trade_result,
                    self.data_fetcher.symbol,
                    self.data_fetcher.timeframe,
                    execution_mode,
                    logger=self.logger,
                )

                try:
                    self.notifier.notify_exit(
                        {**trade_result, 'symbol': self.data_fetcher.symbol}
                    )
                except Exception as notify_err:
                    self.logger.warning(
                        "state=NOTIFY_FAIL action=notify_exit error=%s", notify_err
                    )

                exit_state = "PAPER_EXIT" if self.paper_trading else "LIVE_EXIT"
                self.logger.info(
                    "symbol=%s timeframe=%s state=%s exit_price=%.2f "
                    "pnl=%.2f pnl_percent=%.2f exit_reason=%s",
                    self.data_fetcher.symbol, self.data_fetcher.timeframe,
                    exit_state, exit_price, pnl, pnl_percent,
                    'stop_loss' if stop_loss_hit else 'signal'
                )

                # Clear position
                self.current_position = None

        except Exception as e:
            exit_state = "PAPER_EXIT" if self.paper_trading else "LIVE_EXIT"
            self.logger.error(
                "symbol=%s timeframe=%s state=%s error: %s",
                self.data_fetcher.symbol, self.data_fetcher.timeframe,
                exit_state, e
            )

    def monitor_position(self):
        """Monitor current position and update status."""
        if self.current_position is None:
            return
        
        try:
            status = self.strategy.get_position_status(
                self.market_data,
                self.current_position['entry_price'],
                self.current_position['stop_loss']
            )
            
            if status.get('stop_loss_hit') or status.get('exit_signal'):
                self.check_and_exit_position()
            else:
                self.logger.debug(
                    f"Position status: P&L={status.get('pnl_percent', 0):.2f}%, "
                    f"Price={status.get('current_price', 0):.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Error monitoring position: {e}")

    def run(self):
        """Main bot loop."""
        self.running = True
        self.logger.info("Trading bot started")
        
        try:
            while self.running:
                try:
                    # Update market data
                    if not self.update_market_data():
                        self.logger.warning(
                            "symbol=%s timeframe=%s state=HALT "
                            "reason=update_market_data_failed retrying_in=60s",
                            self.data_fetcher.symbol, self.data_fetcher.timeframe
                        )
                        time.sleep(60)  # Wait 1 minute before retry
                        continue
                    
                    # Monitor existing position
                    if self.current_position is not None:
                        self.monitor_position()
                    else:
                        # Check for entry signals
                        self.check_and_enter_position()
                    
                    # Check kill switch
                    kill_switch, _ = self.risk_manager.check_kill_switch(
                        self.data_fetcher.consecutive_errors,
                        consecutive_none_market_data=self.consecutive_none_market_data,
                        consecutive_price_failures=self.consecutive_none_price
                    )
                    if self.risk_manager.kill_switch_active or kill_switch:
                        self.logger.critical(
                            "symbol=%s timeframe=%s state=HALT "
                            "kill_switch_active reason=%s",
                            self.data_fetcher.symbol, self.data_fetcher.timeframe,
                            self.risk_manager.kill_switch_reason
                        )
                        try:
                            self.notifier.notify_kill_switch(
                                self.risk_manager.kill_switch_reason or "Unknown"
                            )
                        except Exception as notify_err:
                            self.logger.warning(
                                "state=NOTIFY_FAIL action=notify_kill_switch error=%s",
                                notify_err,
                            )
                        if self.current_position is not None:
                            exit_state = (
                                "PAPER_EXIT" if self.paper_trading else "LIVE_EXIT"
                            )
                            self.logger.info(
                                "symbol=%s timeframe=%s state=%s "
                                "reason=closing_position_due_to_kill_switch",
                                self.data_fetcher.symbol, self.data_fetcher.timeframe,
                                exit_state
                            )
                            self.check_and_exit_position()
                        self.running = False
                        break

                    # Log status periodically
                    risk_status = self.risk_manager.get_risk_status()
                    self.logger.info(
                        "symbol=%s timeframe=%s state=MONITOR capital=%.2f "
                        "daily_pnl=%.2f daily_trades=%d",
                        self.data_fetcher.symbol, self.data_fetcher.timeframe,
                        risk_status['current_capital'], risk_status['daily_pnl'],
                        risk_status['daily_trades_count']
                    )

                    # Optional: send status to Telegram at most once per hour
                    now_ts = time.time()
                    if now_ts - self.last_status_notify_time >= 3600:
                        try:
                            self.notifier.notify_status(risk_status)
                            self.last_status_notify_time = now_ts
                        except Exception as notify_err:
                            self.logger.warning(
                                "state=NOTIFY_FAIL action=notify_status error=%s",
                                notify_err,
                            )

                    # Wait for next cycle (4-hour timeframe, check every 5 minutes)
                    time.sleep(300)  # 5 minutes
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal, shutting down...")
                    self.running = False
                    break
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait before retrying
                    
        finally:
            self.logger.info("Trading bot stopped")
            if self.current_position is not None:
                self.logger.warning("Bot stopped with open position!")


def main():
    """Main entry point."""
    bot = TradingBot()
    bot.run()


if __name__ == '__main__':
    main()
