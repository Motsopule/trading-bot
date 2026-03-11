"""
Risk management module for controlling trading risk and enforcing limits.

This module handles daily loss limits, trading hours, kill switch logic,
and position sizing.
"""

import logging
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone
import json
import os

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages trading risk and enforces risk limits.
    
    Features:
    - Daily loss limit tracking
    - Trading hours enforcement (London/NY overlap)
    - Kill switch for consecutive errors or large price moves
    - Position sizing
    """
    
    def __init__(
        self,
        initial_capital: float,
        daily_loss_limit_percent: float = 3.0,
        trading_start_hour: int = 8,
        trading_end_hour: int = 17,
        max_consecutive_errors: int = 5,
        max_price_move_percent: float = 10.0,
        max_consecutive_none_market_data: int = 3,
        max_consecutive_price_failures: int = 3,
        state_file: str = 'risk_state.json'
    ):
        """
        Initialize the RiskManager.

        Args:
            initial_capital: Starting capital amount
            daily_loss_limit_percent: Maximum daily loss as percentage (default: 3.0)
            trading_start_hour: Trading start hour in UTC (default: 8)
            trading_end_hour: Trading end hour in UTC (default: 17)
            max_consecutive_errors: Max API errors before kill switch (default: 5)
            max_price_move_percent: Max price move % before kill switch (default: 10.0)
            max_consecutive_none_market_data: Max consecutive None klines before kill switch (default: 3)
            max_consecutive_price_failures: Max consecutive price fetch failures before kill switch (default: 3)
            state_file: File to persist risk state (default: 'risk_state.json')
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_loss_limit_percent = daily_loss_limit_percent
        # Daily loss limit is computed dynamically from current_capital (see _daily_loss_limit_amount)

        self.trading_start_hour = trading_start_hour
        self.trading_end_hour = trading_end_hour

        self.max_consecutive_errors = max_consecutive_errors
        self.max_price_move_percent = max_price_move_percent
        self.max_consecutive_none_market_data = max_consecutive_none_market_data
        self.max_consecutive_price_failures = max_consecutive_price_failures
        
        self.state_file = state_file
        
        # Daily tracking
        self.current_date = None
        self.daily_pnl = 0.0
        self.daily_trades = []
        
        # Kill switch state
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        # Load state if exists
        self.load_state()
        
        self.max_position_exposure_percent = 50.0  # Max position value = 50% of capital

        logger.info(
            f"RiskManager initialized: capital={initial_capital}, "
            f"daily_loss_limit={daily_loss_limit_percent}% of current capital, "
            f"max_position_exposure={self.max_position_exposure_percent}%, "
            f"trading_hours={trading_start_hour}-{trading_end_hour} UTC"
        )

    def _daily_loss_limit_amount(self) -> float:
        """Dynamic daily loss limit based on current capital."""
        return self.current_capital * (self.daily_loss_limit_percent / 100.0)

    def reset_daily_tracking(self):
        """Reset daily tracking for a new day."""
        today = datetime.now(timezone.utc).date()
        
        if self.current_date != today:
            logger.info(f"Resetting daily tracking for {today}")
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades = []
            self.save_state()

    def is_within_trading_hours(self) -> bool:
        """
        Check if current time is within trading hours (London/NY overlap).
        
        Returns:
            True if within trading hours, False otherwise
        """
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        
        is_within = self.trading_start_hour <= current_hour < self.trading_end_hour
        
        if not is_within:
            logger.debug(
                f"Outside trading hours: {current_hour} UTC "
                f"(allowed: {self.trading_start_hour}-{self.trading_end_hour})"
            )
        
        return is_within

    def can_open_new_trade(self) -> Tuple[bool, Optional[str]]:
        """
        Check if a new trade can be opened.
        
        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        # Reset daily tracking if needed
        self.reset_daily_tracking()
        
        # Check kill switch
        if self.kill_switch_active:
            return False, f"Kill switch active: {self.kill_switch_reason}"
        
        # Check trading hours
        if not self.is_within_trading_hours():
            return False, "Outside trading hours"
        
        # Check daily loss limit (dynamic from current capital)
        limit = self._daily_loss_limit_amount()
        if self.daily_pnl <= -limit:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f} (limit: {limit:.2f})"
        
        return True, None

    def can_add_position(
        self,
        current_total_exposure: float,
        new_position_value: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a new position would exceed max portfolio exposure (50%).
        Used for multi-asset trading: total exposure across all open positions
        must not exceed max_position_exposure_percent of capital.

        Args:
            current_total_exposure: Sum of (position value) for all open positions.
            new_position_value: Notional value of the new position to open.

        Returns:
            (True, None) if allowed; (False, reason) if would exceed limit.
        """
        max_exposure = self.current_capital * (
            self.max_position_exposure_percent / 100.0
        )
        total_after = current_total_exposure + new_position_value
        if total_after > max_exposure:
            return (
                False,
                f"Max portfolio exposure would be exceeded: "
                f"{total_after:.2f} > {max_exposure:.2f} ({self.max_position_exposure_percent}% of capital)",
            )
        return True, None

    def record_trade(self, trade_result: Dict):
        """
        Record a completed trade and update daily P&L.
        
        Args:
            trade_result: Dictionary with trade information including 'pnl'
        """
        self.reset_daily_tracking()
        
        pnl = trade_result.get('pnl', 0.0)
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        trade_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'pnl': pnl,
            'daily_pnl': self.daily_pnl,
            'capital': self.current_capital
        }
        trade_record.update(trade_result)
        
        self.daily_trades.append(trade_record)
        
        logger.info(
            f"Trade recorded: P&L={pnl:.2f}, Daily P&L={self.daily_pnl:.2f}, "
            f"Capital={self.current_capital:.2f}"
        )
        
        # Check if daily loss limit is hit (dynamic)
        limit = self._daily_loss_limit_amount()
        if self.daily_pnl <= -limit:
            logger.warning(
                f"Daily loss limit reached: {self.daily_pnl:.2f} "
                f"(limit: {limit:.2f})"
            )
        
        self.save_state()

    def check_kill_switch(
        self,
        consecutive_errors: int,
        price_change_percent: Optional[float] = None,
        consecutive_none_market_data: int = 0,
        consecutive_price_failures: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if kill switch should be activated.

        Args:
            consecutive_errors: Number of consecutive API errors
            price_change_percent: Recent price change percentage (optional)
            consecutive_none_market_data: Consecutive cycles with None klines (optional)
            consecutive_price_failures: Consecutive price fetch failures (optional)

        Returns:
            Tuple of (should_activate, reason)
        """
        # Check consecutive API errors
        if consecutive_errors >= self.max_consecutive_errors:
            reason = f"Too many consecutive API errors: {consecutive_errors}"
            self.activate_kill_switch(reason)
            return True, reason

        # Check repeated None market data
        if consecutive_none_market_data >= self.max_consecutive_none_market_data:
            reason = (
                f"Repeated None market data: {consecutive_none_market_data} "
                f"consecutive kline fetch failures"
            )
            self.activate_kill_switch(reason)
            return True, reason

        # Check repeated price fetch failures
        if consecutive_price_failures >= self.max_consecutive_price_failures:
            reason = (
                f"Repeated price fetch failures: {consecutive_price_failures} "
                f"consecutive price fetch failures"
            )
            self.activate_kill_switch(reason)
            return True, reason

        # Check large price move
        if price_change_percent is not None:
            abs_change = abs(price_change_percent)
            if abs_change >= self.max_price_move_percent:
                reason = f"Large price move detected: {price_change_percent:.2f}%"
                self.activate_kill_switch(reason)
                return True, reason

        return False, None

    def activate_kill_switch(self, reason: str):
        """
        Activate the kill switch.
        
        Args:
            reason: Reason for activating kill switch
        """
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_switch_reason = reason
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
            self.save_state()

    def deactivate_kill_switch(self):
        """Deactivate the kill switch."""
        if self.kill_switch_active:
            logger.info("Kill switch deactivated")
            self.kill_switch_active = False
            self.kill_switch_reason = None
            self.save_state()

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 1.0
    ) -> float:
        """
        Calculate position size based on risk percentage, capped by max position
        exposure (default 50% of capital).
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop-loss price
            risk_percent: Percentage of capital to risk (default: 1.0)
            
        Returns:
            Position size in base asset units
        """
        if entry_price <= 0 or stop_loss <= 0:
            logger.warning("Invalid prices for position sizing")
            return 0.0
        
        risk_per_trade = self.current_capital * (risk_percent / 100.0)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            logger.warning("Zero price risk, cannot calculate position size")
            return 0.0
        
        position_size = risk_per_trade / price_risk
        position_value = position_size * entry_price
        max_position_value = self.current_capital * (self.max_position_exposure_percent / 100.0)

        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            logger.debug(
                f"Position size capped by exposure limit: {position_size:.6f} "
                f"(max {self.max_position_exposure_percent}% of capital)"
            )
        else:
            logger.debug(
                f"Position size calculated: {position_size:.6f} "
                f"(risk: {risk_percent}%, capital: {self.current_capital:.2f})"
            )
        
        return position_size

    def get_risk_status(self) -> Dict:
        """
        Get current risk management status.
        
        Returns:
            Dictionary with risk status information
        """
        self.reset_daily_tracking()
        
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self._daily_loss_limit_amount(),
            'daily_loss_limit_percent': self.daily_loss_limit_percent,
            'daily_trades_count': len(self.daily_trades),
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'within_trading_hours': self.is_within_trading_hours(),
            'can_trade': self.can_open_new_trade()[0],
            'current_date': str(self.current_date) if self.current_date else None
        }

    def save_state(self):
        """Save risk state to file."""
        try:
            state = {
                'current_capital': self.current_capital,
                'current_date': str(self.current_date) if self.current_date else None,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades[-10:],  # Keep last 10 trades
                'kill_switch_active': self.kill_switch_active,
                'kill_switch_reason': self.kill_switch_reason
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Risk state saved")
            
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")

    def load_state(self):
        """Load risk state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.current_capital = state.get('current_capital', self.initial_capital)
                date_str = state.get('current_date')
                if date_str:
                    self.current_date = datetime.fromisoformat(date_str).date()
                self.daily_pnl = state.get('daily_pnl', 0.0)
                self.daily_trades = state.get('daily_trades', [])
                self.kill_switch_active = state.get('kill_switch_active', False)
                self.kill_switch_reason = state.get('kill_switch_reason')
                
                logger.info("Risk state loaded from file")
                
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
