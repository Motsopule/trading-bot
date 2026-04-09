"""
Main trading bot module that orchestrates data fetching, strategy,
risk management, and order execution.

This is the entry point for the crypto trading bot.
"""

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from dotenv import load_dotenv

from data import DataFetcher
from strategy import TradingStrategy
from risk import PortfolioRiskEngine, PortfolioStateBuilder, KillSwitch
from risk.models import RiskDecision, Signal
from risk.sizing import PositionSizer
from portfolio import PortfolioAllocator, build_portfolio_context
from intelligence.portfolio_optimizer import PortfolioOptimizer
from intelligence.models import build_returns_matrix
from risk_manager import RiskManager
from execution import (
    OrderExecutor,
    OMS,
    ExecutionEngine,
    MultiSymbolExchangeAdapter,
    FillHandler,
    PositionManager,
    ReconciliationEngine,
)
from execution.order_model import OrderStatus
from journal import write_trade_entry
from notifications import TelegramNotifier
from equity import log_equity
from strategy_control.asset_classifier import get_asset_class
from strategy_control.filters import apply_filters
from strategy_control.performance_tracker import PerformanceTracker
from strategy_control.regime_detector import RegimeDetector
from strategy_control.strategy_context import make_strategy_context
from strategy_control.strategy_registry import load_strategy
from strategy_control.strategy_router import StrategyRouter

# Telegram status message throttle: send at most every 4 hours
STATUS_INTERVAL = timedelta(hours=4)

# Market scanner interval: run every 4 hours when enabled
SCAN_INTERVAL = timedelta(hours=4)

# Strategy control: max strategies that may place orders per symbol per evaluation
MAX_STRATEGIES_PER_SYMBOL = 1

_INDICATOR_KEYS_REQUIRED = ("ma50", "ma200", "atr")

# Load environment variables from project directory (reliable regardless of cwd)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path)


def _mask_key(value: str) -> str:
    """Mask a key for debug output: show first and last char only."""
    if not value or not value.strip():
        return "<empty>"
    v = value.strip()
    if len(v) <= 4:
        return "*" * len(v)
    return f"{v[0]}***{v[-1]}"


# Debug: print whether Binance env vars are loaded (masked)
_env_api_key = os.getenv("BINANCE_API_KEY", "").strip()
_env_api_secret = (os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET") or "").strip()
_env_testnet = os.getenv("BINANCE_TESTNET", "True").strip().lower() == "true"
print(
    "[Binance env] BINANCE_API_KEY=%s | BINANCE_SECRET_KEY/API_SECRET=%s | "
    "BINANCE_TESTNET=%s -> testnet_mode=%s"
    % (_mask_key(_env_api_key), _mask_key(_env_api_secret), os.getenv("BINANCE_TESTNET", ""), _env_testnet)
)

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
        # Support both BINANCE_SECRET_KEY (e.g. .env) and BINANCE_API_SECRET (docs)
        api_key = (os.getenv('BINANCE_API_KEY') or "").strip()
        api_secret = (
            (os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_API_SECRET')) or ""
        ).strip()
        testnet = os.getenv('BINANCE_TESTNET', 'True').strip().lower() == 'true'

        if not testnet:
            self.logger.error("Live trading attempt detected. Testnet mode is required.")
            raise RuntimeError("Live trading is disabled. This bot may only run in Testnet mode.")

        # Multi-asset: TRADING_PAIRS comma-separated, or single TRADING_PAIR
        _pairs_env = (os.getenv('TRADING_PAIRS') or '').strip()
        if _pairs_env:
            self.trading_pairs: List[str] = [
                s.strip() for s in _pairs_env.split(',') if s.strip()
            ]
        else:
            self.trading_pairs = [os.getenv('TRADING_PAIR', 'ETHUSDT').strip()]
        if not self.trading_pairs:
            self.logger.error("No trading pairs configured (TRADING_PAIRS or TRADING_PAIR)")
            sys.exit(1)
        self._symbol_index_map = {s: i for i, s in enumerate(self.trading_pairs)}
        trading_pair = self.trading_pairs[0]  # default for initial data_fetcher
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
        
        # One executor per symbol for multi-asset
        self.executors: Dict[str, OrderExecutor] = {
            sym: OrderExecutor(
                self.data_fetcher.client,
                sym,
                paper_trading=paper_trading,
            )
            for sym in self.trading_pairs
        }
        self.paper_trading = paper_trading

        self.oms = OMS()
        self.fill_handler = FillHandler()
        self.oms_position_manager = PositionManager()
        self.reconciliation_engine = ReconciliationEngine()
        self.exchange_adapter = MultiSymbolExchangeAdapter(self.executors)
        self.execution_engine = ExecutionEngine(self.exchange_adapter)
        self._oms_order_timeout_seconds = float(
            os.getenv("OMS_ORDER_TIMEOUT_SECONDS", "0").strip() or "0"
        )

        self.portfolio_risk_engine = PortfolioRiskEngine()
        self.portfolio_state_builder = PortfolioStateBuilder(
            self.data_fetcher.client,
            symbols=self.trading_pairs,
            position_enricher=self._position_enricher_for_risk,
        )

        self.notifier = TelegramNotifier()

        self.regime_detector = RegimeDetector()
        self.strategy_router = StrategyRouter()
        self.performance_tracker = PerformanceTracker()
        self.portfolio_allocator = PortfolioAllocator()
        self.portfolio_optimizer = PortfolioOptimizer()

        # Bot state (multi-asset: positions and market_data keyed by symbol)
        self.running = False
        self.positions: Dict[str, Dict] = {}  # symbol -> position dict
        self.market_data: Dict[str, object] = {}  # symbol -> DataFrame
        self.last_price: Dict[str, float] = {}  # symbol -> last price
        self.price_history: List[float] = []  # global for kill-switch volatility
        self.consecutive_none_market_data = 0
        self.consecutive_none_price = 0
        self.last_status_time = datetime.now(timezone.utc) - timedelta(hours=4)

        # Market scanner scheduling (non-intrusive to trading loop)
        self.enable_market_scanner = (
            os.getenv("ENABLE_MARKET_SCANNER", "").strip().lower() == "true"
        )
        self.last_scan_time = datetime.now(timezone.utc) - SCAN_INTERVAL

        self.logger.info("Trading bot initialized successfully")
        self.logger.info(
            "Trading pairs: %s, Timeframe: %s",
            self.trading_pairs,
            timeframe,
        )
        self.logger.info(f"Testnet mode: {testnet}")
        self.logger.info(
            "Execution mode: %s",
            "PAPER (simulated)" if paper_trading else "LIVE (real orders)"
        )

    def _position_enricher_for_risk(self) -> Dict[str, dict]:
        """Entry/stop for open positions (exchange remains size/equity source of truth)."""
        out: Dict[str, dict] = {}
        for sym, pos in self.positions.items():
            out[sym] = {
                "entry_price": float(pos["entry_price"]),
                "stop_loss": float(pos["stop_loss"]),
                "side": "LONG",
            }
        return out

    def _log_risk_decision(self, signal: Signal, decision: RiskDecision) -> None:
        try:
            payload = {
                "symbol": signal.symbol,
                "decision": bool(decision.allowed),
                "reason": decision.reason,
                "portfolio_risk": float(decision.portfolio_risk),
                "proposed_risk": float(decision.proposed_trade_risk),
            }
            self.logger.info("risk_decision %s", json.dumps(payload))
        except Exception:
            pass

    def _log_strategy_control(
        self,
        event: str,
        symbol: str,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
        **extra,
    ) -> None:
        """Structured strategy_control logs; must never break execution."""
        payload = {
            "event": event,
            "symbol": symbol,
            "regime": regime,
            "strategy": strategy,
            **extra,
        }
        try:
            self.logger.info("strategy_control %s", json.dumps(payload))
        except Exception:
            pass

    @staticmethod
    def _indicators_from_df(df) -> Optional[Dict]:
        """Build indicator dict for regime detection and filters from last closed bar."""
        if df is None or len(df) < 1:
            return None
        row = df.iloc[-1]

        def _f(x) -> float:
            if x is None:
                return 0.0
            try:
                v = float(x)
                return 0.0 if math.isnan(v) else v
            except (TypeError, ValueError):
                return 0.0

        atr_ma = _f(row.get("atr_ma"))
        # Phase 5: same numeric baseline as atr_threshold (strategy atr_ma); set once per bar
        atr_baseline = atr_ma
        return {
            "ma50": _f(row.get("ma_50")),
            "ma200": _f(row.get("ma_200")),
            "atr": _f(row.get("atr")),
            "atr_threshold": atr_ma,
            "atr_baseline": float(atr_baseline),
            "min_atr": atr_ma,
        }

    @staticmethod
    def _indicators_complete(indicators: Dict) -> Optional[str]:
        """Return missing key if any required indicator is absent; else None."""
        for key in _INDICATOR_KEYS_REQUIRED:
            if key not in indicators:
                return key
        return None

    def update_market_data(self, symbol: str) -> bool:
        """
        Fetch and update market data for one symbol.

        Validates candle count against required MA period; skips indicator
        calculation if insufficient data.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.data_fetcher.set_trading_pair(symbol, self.data_fetcher.timeframe or '4h')
            timeframe = self.data_fetcher.timeframe
            required_ma_period = self.strategy.ma_200_period

            df = self.data_fetcher.get_klines(limit=500)
            if df is None:
                self.logger.warning(
                    "symbol=%s timeframe=%s state=HALT reason=market_data_none "
                    "consecutive_none_klines=%d",
                    symbol, timeframe, self.consecutive_none_market_data
                )
                return False

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
                self.logger.warning(
                    "symbol=%s timeframe=%s state=HALT reason=price_fetch_failed",
                    symbol, timeframe
                )
                return False
            
            # Track price history for volatility check (global)
            prev = self.last_price.get(symbol)
            if prev is not None:
                price_change = ((current_price - prev) / prev) * 100
                self.price_history.append(price_change)
                if len(self.price_history) > 10:
                    self.price_history.pop(0)
                if abs(price_change) >= self.risk_manager.max_price_move_percent:
                    self.risk_manager.check_kill_switch(
                        self.data_fetcher.consecutive_errors,
                        price_change_percent=price_change,
                        consecutive_none_market_data=self.consecutive_none_market_data,
                        consecutive_price_failures=self.consecutive_none_price
                    )
            self.last_price[symbol] = current_price
            self.market_data[symbol] = df
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating market data for {symbol}: {e}")
            return False

    def _total_exposure_usdt(self) -> float:
        """Sum of (position value) across all open positions using latest market data."""
        total = 0.0
        for sym, pos in self.positions.items():
            df = self.market_data.get(sym)
            if df is not None and len(df) > 0 and pos.get('quantity'):
                price = float(df.iloc[-1]['close'])
                total += pos['quantity'] * price
        return total

    def check_and_enter_position(self, symbol: str, total_exposure: float):
        """Check for entry signals and enter position if conditions are met for symbol."""
        if self.positions.get(symbol) is not None:
            return  # Already in a position for this symbol
        
        df = self.market_data.get(symbol)
        if df is None:
            return
        
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
                "symbol=%s state=HALT kill_switch_check_failed: %s",
                symbol, kill_reason
            )
            return

        indicators = self._indicators_from_df(df)
        if indicators is None:
            return

        missing = self._indicators_complete(indicators)
        if missing is not None:
            self._log_strategy_control(
                "missing_indicator",
                symbol,
                regime=None,
                strategy=None,
                missing=missing,
            )
            return

        asset_class = get_asset_class(symbol)
        if asset_class == "unknown":
            self._log_strategy_control(
                "unknown_asset_class",
                symbol,
                regime=None,
                strategy=None,
            )
            return

        regime = self.regime_detector.detect(symbol, indicators)
        self._log_strategy_control(
            "regime_detected",
            symbol,
            regime=regime,
            strategy=None,
        )

        strategies = self.strategy_router.get_strategies(
            asset_class, regime, symbol=symbol
        )

        if not strategies:
            self._log_strategy_control(
                event="no_strategy",
                symbol=symbol,
                regime=regime,
                strategy=None,
                asset_class=asset_class,
            )
            return

        price_data = {"df": df, "data_client": self.data_fetcher}
        executed = 0

        for strategy_name in strategies:
            if executed >= MAX_STRATEGIES_PER_SYMBOL:
                break

            self._log_strategy_control(
                event="strategy_attempt",
                symbol=symbol,
                regime=regime,
                strategy=strategy_name,
            )

            strategy_impl = load_strategy(strategy_name, self.strategy)
            if strategy_impl is None:
                self._log_strategy_control(
                    event="no_signal",
                    symbol=symbol,
                    regime=regime,
                    strategy=strategy_name,
                )
                continue

            context = make_strategy_context(
                symbol,
                asset_class,
                regime,
                indicators,
                price_data,
            )
            signal = strategy_impl.generate(context)
            if not signal:
                self._log_strategy_control(
                    event="no_signal",
                    symbol=symbol,
                    regime=regime,
                    strategy=strategy_name,
                )
                continue
            if not apply_filters(signal, context):
                self._log_strategy_control(
                    event="filter_blocked",
                    symbol=symbol,
                    regime=regime,
                    strategy=strategy_name,
                )
                continue

            self._log_strategy_control(
                event="signal_detected",
                symbol=symbol,
                regime=regime,
                strategy=strategy_name,
                asset_class=asset_class,
            )

            signal_details = signal["signal_details"]
            strategy_name_used = strategy_name

            try:
                stop_loss = self.strategy.calculate_stop_loss(
                    signal_details['entry_price'],
                    signal_details['atr']
                )

                if KillSwitch.is_active():
                    self.logger.warning("BLOCKED: Kill switch active")
                    return

                portfolio = self.portfolio_state_builder.build()
                sig = Signal(
                    symbol=symbol,
                    side="LONG",
                    entry_price=float(signal_details["entry_price"]),
                    stop_loss=float(stop_loss),
                )
                strategy_by_symbol = {
                    sym: str(pos.get("strategy_name", "unknown"))
                    for sym, pos in self.positions.items()
                }
                portfolio_context = build_portfolio_context(
                    portfolio,
                    strategy_by_symbol,
                    self.performance_tracker.get_all_stats(),
                )
                base_size = PositionSizer.position_size_from_risk(
                    sig,
                    PositionSizer.calculate(sig, portfolio),
                )
                adjusted_size = self.portfolio_allocator.adjust_size(
                    strategy_name_used,
                    symbol,
                    base_size,
                    portfolio_context,
                )
                if adjusted_size <= 0:
                    try:
                        self.logger.info(
                            "allocation_blocked %s",
                            json.dumps(
                                {
                                    "event": "allocation_blocked",
                                    "strategy": strategy_name_used,
                                    "symbol": symbol,
                                }
                            ),
                        )
                    except Exception:
                        self.logger.info(
                            "allocation_blocked strategy=%s symbol=%s",
                            strategy_name_used,
                            symbol,
                        )
                    continue

                returns_matrix = build_returns_matrix(
                    self.trading_pairs, self.market_data
                )
                symbol_idx = self._symbol_index_map[symbol]
                stats = self.performance_tracker.get_strategy_stats(strategy_name_used)
                opt_context = {
                    "strategy": strategy_name_used,
                    "regime": regime,
                    "stats": stats if stats else {},
                    "atr": float(indicators.get("atr", 0)),
                    "atr_baseline": float(indicators.get("atr_baseline", indicators.get("atr_threshold", 0))),
                    "returns_matrix": returns_matrix,
                    "symbol_idx": symbol_idx,
                }
                optimized_size = self.portfolio_optimizer.optimize(
                    adjusted_size, opt_context
                )
                if optimized_size <= 0:
                    continue

                account_state = self.risk_manager.get_account_risk_state(portfolio.equity)
                decision = self.portfolio_risk_engine.evaluate(
                    sig, portfolio, account_state, size_override=optimized_size
                )
                self._log_risk_decision(sig, decision)
                if not decision.allowed:
                    self.logger.warning("BLOCKED: %s", decision.reason)
                    continue

                position_size = decision.size

                if position_size <= 0:
                    self.logger.warning("Invalid position size from risk engine")
                    continue

                required_usdt = position_size * signal_details['entry_price']
                can_add, exposure_reason = self.risk_manager.can_add_position(
                    total_exposure, required_usdt
                )
                if not can_add:
                    self.logger.debug(
                        "symbol=%s skip_entry: %s", symbol, exposure_reason
                    )
                    continue

                if self.paper_trading:
                    usdt_balance = self.risk_manager.get_risk_status()['current_capital']
                else:
                    self.data_fetcher.set_trading_pair(symbol, self.data_fetcher.timeframe or '4h')
                    balances = self.data_fetcher.get_account_balance()
                    if balances is None:
                        continue
                    usdt_balance = balances.get('USDT', 0.0)

                if required_usdt > usdt_balance:
                    self.logger.warning(
                        "symbol=%s insufficient balance: need %.2f, have %.2f",
                        symbol, required_usdt, usdt_balance
                    )
                    continue

                order = self.oms.create_order(symbol, "BUY", position_size, "MARKET")
                if not order:
                    self.logger.info("symbol=%s Duplicate order blocked", symbol)
                    continue

                result = self.execution_engine.submit_order(order)
                if not result.get("success"):
                    self.oms.mark_rejected(order)
                    self.logger.error(
                        "symbol=%s failed to place buy order: %s",
                        symbol,
                        result.get("error"),
                    )
                    continue

                ex_id = result.get("exchange_order_id")
                if not ex_id:
                    self.oms.mark_rejected(order)
                    self.logger.error("symbol=%s buy order missing exchange id", symbol)
                    continue

                raw = result.get("raw") or {}
                order["exchange_order_id"] = ex_id
                order["updated_at"] = datetime.now(timezone.utc).isoformat()
                qty = float(raw.get("quantity", 0) or 0)
                px = raw.get("price")
                prev_filled = float(order.get("filled_quantity", 0) or 0)
                if qty > 0 and px is not None:
                    self.fill_handler.process_fill(
                        order,
                        qty,
                        float(px),
                        persist_after=self.oms._persist,
                    )
                    new_filled = float(order.get("filled_quantity", 0) or 0)
                    self.oms_position_manager.apply_fill_delta(
                        symbol, order["side"], prev_filled, new_filled
                    )
                else:
                    order["status"] = OrderStatus.SUBMITTED.value
                    order["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.oms._persist()

                buy_order = raw
                entry_price = (
                    float(order["price"])
                    if order.get("price") is not None
                    else (float(px) if px is not None else signal_details["entry_price"])
                )
                executed_qty = float(order.get("filled_quantity", 0) or 0)
                if executed_qty <= 0:
                    self.logger.error(
                        "symbol=%s buy order returned no executed quantity", symbol
                    )
                    continue

                self.positions[symbol] = {
                    'entry_price': entry_price,
                    'entry_time': datetime.now(timezone.utc),
                    'quantity': executed_qty,
                    'stop_loss': stop_loss,
                    'entry_signal': signal_details,
                    'buy_order': buy_order,
                    'strategy_name': strategy_name_used,
                    'entry_regime': regime,
                }

                executor = self.executors[symbol]
                stop_order = executor.place_stop_loss_order(executed_qty, stop_loss)
                if stop_order:
                    self.positions[symbol]['stop_order'] = stop_order

                timeframe = self.data_fetcher.timeframe or '4h'
                entry_state = "PAPER_ENTRY" if self.paper_trading else "LIVE_ENTRY"
                self.logger.info(
                    "symbol=%s timeframe=%s state=%s executed_qty=%.6f "
                    "entry_price=%.2f stop_loss=%.2f",
                    symbol, timeframe, entry_state, executed_qty, entry_price, stop_loss
                )

                try:
                    self.notifier.notify_entry(
                        {**self.positions[symbol], 'symbol': symbol}
                    )
                except Exception as notify_err:
                    self.logger.warning(
                        "state=NOTIFY_FAIL action=notify_entry error=%s", notify_err
                    )

                executed += 1

            except Exception as e:
                entry_state = "PAPER_ENTRY" if self.paper_trading else "LIVE_ENTRY"
                self.logger.critical(
                    "CRITICAL ERROR symbol=%s state=%s: %s", symbol, entry_state, e
                )
                try:
                    KillSwitch.trigger("System failure")
                except Exception:
                    pass
                return

    def check_and_exit_position(self, symbol: str):
        """Check for exit signals and exit position if conditions are met for symbol."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        
        df = self.market_data.get(symbol)
        if df is None:
            return
        
        try:
            entry_price = pos['entry_price']
            exit_signal, exit_details = self.strategy.check_exit_signal(df, entry_price)
            
            self.data_fetcher.set_trading_pair(symbol, self.data_fetcher.timeframe or '4h')
            current_price = self.data_fetcher.get_current_price()
            if current_price is None and len(df) > 0:
                current_price = float(df.iloc[-1]['close'])
            if current_price is None:
                return
            
            stop_loss_hit = self.strategy.check_stop_loss(
                current_price, pos['stop_loss']
            )
            
            if exit_signal or stop_loss_hit:
                quantity = pos['quantity']
                order = self.oms.create_order(symbol, "SELL", quantity, "MARKET")
                if not order:
                    self.logger.info("symbol=%s Duplicate order blocked (exit)", symbol)
                    return

                result = self.execution_engine.submit_order(order)
                if not result.get("success"):
                    self.oms.mark_rejected(order)
                    self.logger.error(
                        "symbol=%s failed to place sell order: %s",
                        symbol,
                        result.get("error"),
                    )
                    return

                ex_id = result.get("exchange_order_id")
                if not ex_id:
                    self.oms.mark_rejected(order)
                    self.logger.error("symbol=%s sell order missing exchange id", symbol)
                    return

                raw = result.get("raw") or {}
                order["exchange_order_id"] = ex_id
                order["updated_at"] = datetime.now(timezone.utc).isoformat()
                qty = float(raw.get("quantity", 0) or 0)
                px = raw.get("price")
                prev_filled = float(order.get("filled_quantity", 0) or 0)
                if qty > 0 and px is not None:
                    self.fill_handler.process_fill(
                        order,
                        qty,
                        float(px),
                        persist_after=self.oms._persist,
                    )
                    new_filled = float(order.get("filled_quantity", 0) or 0)
                    self.oms_position_manager.apply_fill_delta(
                        symbol, order["side"], prev_filled, new_filled
                    )
                else:
                    order["status"] = OrderStatus.SUBMITTED.value
                    order["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.oms._persist()

                sell_order = raw
                exit_price = (
                    float(order["price"])
                    if order.get("price") is not None
                    else (float(px) if px is not None else current_price)
                )
                executed_sell_qty = float(order.get("filled_quantity", 0) or 0)
                if executed_sell_qty <= 0:
                    self.logger.error(
                        "symbol=%s sell order returned no executed quantity", symbol
                    )
                    return
                pnl = (exit_price - entry_price) * executed_sell_qty
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100

                self.performance_tracker.log_trade(
                    pos.get("strategy_name", "unknown"),
                    pnl,
                    symbol,
                    pos.get("entry_regime", "unknown"),
                )

                trade_result = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': executed_sell_qty,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'entry_time': pos['entry_time'].isoformat(),
                    'exit_time': datetime.now(timezone.utc).isoformat(),
                    'exit_reason': 'stop_loss' if stop_loss_hit else 'signal',
                    'sell_order': sell_order
                }
                
                self.risk_manager.record_trade(trade_result)

                execution_mode = "PAPER" if self.paper_trading else "LIVE"
                timeframe = self.data_fetcher.timeframe or '4h'
                write_trade_entry(
                    trade_result,
                    symbol,
                    timeframe,
                    execution_mode,
                    logger=self.logger,
                )

                try:
                    self.notifier.notify_exit({**trade_result, 'symbol': symbol})
                except Exception as notify_err:
                    self.logger.warning(
                        "state=NOTIFY_FAIL action=notify_exit error=%s", notify_err
                    )

                exit_state = "PAPER_EXIT" if self.paper_trading else "LIVE_EXIT"
                self.logger.info(
                    "symbol=%s state=%s exit_price=%.2f pnl=%.2f pnl_percent=%.2f exit_reason=%s",
                    symbol, exit_state, exit_price, pnl, pnl_percent,
                    'stop_loss' if stop_loss_hit else 'signal'
                )

                del self.positions[symbol]

        except Exception as e:
            exit_state = "PAPER_EXIT" if self.paper_trading else "LIVE_EXIT"
            self.logger.error("symbol=%s state=%s error: %s", symbol, exit_state, e)

    def monitor_position(self, symbol: str):
        """Monitor current position and update status for symbol."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        
        df = self.market_data.get(symbol)
        if df is None:
            return
        
        try:
            status = self.strategy.get_position_status(
                df,
                pos['entry_price'],
                pos['stop_loss']
            )
            
            if status.get('stop_loss_hit') or status.get('exit_signal'):
                self.check_and_exit_position(symbol)
            else:
                self.logger.debug(
                    "symbol=%s position P&L=%.2f%% price=%.2f",
                    symbol, status.get('pnl_percent', 0), status.get('current_price', 0)
                )
                
        except Exception as e:
            self.logger.error("symbol=%s error monitoring position: %s", symbol, e)

    def run(self):
        """Main bot loop."""
        self.running = True
        self.logger.info("Trading bot started")

        try:
            while self.running:
                try:
                    self.risk_manager.sync_kill_switch_from_file()

                    if self._oms_order_timeout_seconds > 0:
                        self.oms.cancel_stale_orders(
                            self.exchange_adapter,
                            self._oms_order_timeout_seconds,
                        )
                    self.reconciliation_engine.reconcile(
                        self.oms,
                        self.exchange_adapter,
                        paper_trading=self.paper_trading,
                    )
                    self.oms._persist()

                    # Update market data for all symbols
                    any_update_ok = False
                    for sym in self.trading_pairs:
                        if self.update_market_data(sym):
                            any_update_ok = True
                    if any_update_ok:
                        self.consecutive_none_market_data = 0
                        self.consecutive_none_price = 0
                    else:
                        self.consecutive_none_market_data += 1
                        self.logger.warning(
                            "state=HALT reason=no_market_data_updated "
                            "consecutive=%d retrying_in=60s",
                            self.consecutive_none_market_data,
                        )
                        time.sleep(60)
                        continue
                    
                    total_exposure = self._total_exposure_usdt()
                    
                    for sym in self.trading_pairs:
                        if self.positions.get(sym) is not None:
                            self.monitor_position(sym)
                        else:
                            self.check_and_enter_position(sym, total_exposure)
                            total_exposure = self._total_exposure_usdt()
                    
                    # Check kill switch
                    kill_switch, _ = self.risk_manager.check_kill_switch(
                        self.data_fetcher.consecutive_errors,
                        consecutive_none_market_data=self.consecutive_none_market_data,
                        consecutive_price_failures=self.consecutive_none_price
                    )
                    if self.risk_manager.kill_switch_active or kill_switch:
                        self.logger.critical(
                            "state=HALT kill_switch_active reason=%s",
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
                        for sym in list(self.positions.keys()):
                            self.logger.info(
                                "symbol=%s closing_position_due_to_kill_switch", sym
                            )
                            self.check_and_exit_position(sym)
                        self.running = False
                        break

                    risk_status = self.risk_manager.get_risk_status()
                    self.logger.info(
                        "state=MONITOR capital=%.2f daily_pnl=%.2f daily_trades=%d "
                        "positions=%d exposure=%.2f",
                        risk_status['current_capital'], risk_status['daily_pnl'],
                        risk_status['daily_trades_count'],
                        len(self.positions),
                        total_exposure,
                    )

                    # Periodic market scanner (signals only; does not place trades)
                    if self.enable_market_scanner:
                        try:
                            now = datetime.now(timezone.utc)
                            if now - self.last_scan_time > SCAN_INTERVAL:
                                from scanner import scan_markets
                                candidates = scan_markets(self.strategy, self.data_fetcher)
                                self.logger.info(
                                    "scanner completed: %d signal candidate(s) (trading unchanged)",
                                    len(candidates),
                                )
                                self.last_scan_time = now
                        except Exception as e:
                            self.logger.warning("scanner run failed (trading unchanged): %s", e)

                    log_equity(
                        risk_status['current_capital'],
                        risk_status['daily_pnl'],
                        len(self.positions),
                        total_exposure,
                    )

                    if datetime.now(timezone.utc) - self.last_status_time > STATUS_INTERVAL:
                        try:
                            self.notifier.notify_status(risk_status)
                            self.last_status_time = datetime.now(timezone.utc)
                        except Exception as notify_err:
                            self.logger.warning(
                                "state=NOTIFY_FAIL action=notify_status error=%s",
                                notify_err,
                            )

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
            if self.positions:
                self.logger.warning("Bot stopped with open position(s): %s", list(self.positions.keys()))


def main():
    """Main entry point."""
    bot = TradingBot()
    bot.run()


if __name__ == '__main__':
    main()
