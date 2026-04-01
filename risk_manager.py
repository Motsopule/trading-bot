"""
Risk management module for controlling trading risk and enforcing limits.

This module handles daily loss limits, trading hours, kill switch logic,
and position sizing. Kill-switch activation is mirrored to data/kill_switch.json
for restart-safe gating with the portfolio risk engine.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Persisted start-of-day equity anchor (survives process restarts; UTC calendar date).
DAILY_EQUITY_FILE = os.path.join("data", "daily_equity.json")


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
        state_file: str = "risk_state.json",
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_loss_limit_percent = daily_loss_limit_percent

        self.trading_start_hour = trading_start_hour
        self.trading_end_hour = trading_end_hour

        self.max_consecutive_errors = max_consecutive_errors
        self.max_price_move_percent = max_price_move_percent
        self.max_consecutive_none_market_data = max_consecutive_none_market_data
        self.max_consecutive_price_failures = max_consecutive_price_failures

        self.state_file = state_file

        self.current_date = None
        self.daily_pnl = 0.0
        self.daily_trades = []

        self.kill_switch_active = False
        self.kill_switch_reason = None

        self.peak_capital = initial_capital
        self._risk_peak_equity: Optional[float] = None
        self._start_of_day_equity: Optional[float] = None

        self.load_state()

        self.max_position_exposure_percent = 50.0

        logger.info(
            "RiskManager initialized: capital=%s, "
            "daily_loss_limit=%s%% of current capital, "
            "max_position_exposure=%s%%, "
            "trading_hours=%s-%s UTC",
            initial_capital,
            daily_loss_limit_percent,
            self.max_position_exposure_percent,
            trading_start_hour,
            trading_end_hour,
        )

    def _daily_loss_limit_amount(self) -> float:
        return self.current_capital * (self.daily_loss_limit_percent / 100.0)

    def reset_daily_tracking(self):
        today = datetime.now(timezone.utc).date()

        if self.current_date != today:
            logger.info("Resetting daily tracking for %s", today)
            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades = []
            self._start_of_day_equity = None
            self.save_state()

    def is_within_trading_hours(self) -> bool:
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        is_within = self.trading_start_hour <= current_hour < self.trading_end_hour

        if not is_within:
            logger.debug(
                "Outside trading hours: %s UTC (allowed: %s-%s)",
                current_hour,
                self.trading_start_hour,
                self.trading_end_hour,
            )

        return is_within

    def can_open_new_trade(self) -> Tuple[bool, Optional[str]]:
        self.reset_daily_tracking()

        if self.kill_switch_active:
            return False, f"Kill switch active: {self.kill_switch_reason}"

        if not self.is_within_trading_hours():
            return False, "Outside trading hours"

        limit = self._daily_loss_limit_amount()
        if self.daily_pnl <= -limit:
            return (
                False,
                f"Daily loss limit reached: {self.daily_pnl:.2f} (limit: {limit:.2f})",
            )

        return True, None

    def can_add_position(
        self,
        current_total_exposure: float,
        new_position_value: float,
    ) -> Tuple[bool, Optional[str]]:
        max_exposure = self.current_capital * (self.max_position_exposure_percent / 100.0)
        total_after = current_total_exposure + new_position_value
        if total_after > max_exposure:
            return (
                False,
                f"Max portfolio exposure would be exceeded: "
                f"{total_after:.2f} > {max_exposure:.2f} "
                f"({self.max_position_exposure_percent}% of capital)",
            )
        return True, None

    def record_trade(self, trade_result: Dict):
        self.reset_daily_tracking()

        pnl = trade_result.get("pnl", 0.0)
        self.daily_pnl += pnl
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pnl": pnl,
            "daily_pnl": self.daily_pnl,
            "capital": self.current_capital,
        }
        trade_record.update(trade_result)

        self.daily_trades.append(trade_record)

        logger.info(
            "Trade recorded: P&L=%.2f, Daily P&L=%.2f, Capital=%.2f",
            pnl,
            self.daily_pnl,
            self.current_capital,
        )

        limit = self._daily_loss_limit_amount()
        if self.daily_pnl <= -limit:
            logger.warning(
                "Daily loss limit reached: %s (limit: %s)",
                self.daily_pnl,
                limit,
            )

        self.save_state()

    def check_kill_switch(
        self,
        consecutive_errors: int,
        price_change_percent: Optional[float] = None,
        consecutive_none_market_data: int = 0,
        consecutive_price_failures: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        if consecutive_errors >= self.max_consecutive_errors:
            reason = f"Too many consecutive API errors: {consecutive_errors}"
            self.activate_kill_switch(reason)
            return True, reason

        if consecutive_none_market_data >= self.max_consecutive_none_market_data:
            reason = (
                f"Repeated None market data: {consecutive_none_market_data} "
                "consecutive kline fetch failures"
            )
            self.activate_kill_switch(reason)
            return True, reason

        if consecutive_price_failures >= self.max_consecutive_price_failures:
            reason = (
                f"Repeated price fetch failures: {consecutive_price_failures} "
                "consecutive price fetch failures"
            )
            self.activate_kill_switch(reason)
            return True, reason

        if price_change_percent is not None:
            abs_change = abs(price_change_percent)
            if abs_change >= self.max_price_move_percent:
                reason = f"Large price move detected: {price_change_percent:.2f}%"
                self.activate_kill_switch(reason)
                return True, reason

        return False, None

    def activate_kill_switch(self, reason: str):
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)
        self.save_state()
        try:
            from risk.kill_switch import KillSwitch

            KillSwitch.trigger(reason)
        except Exception as e:
            logger.error("Failed to persist kill switch file: %s", e)

    def deactivate_kill_switch(self):
        if self.kill_switch_active:
            logger.info("Kill switch deactivated")
            self.kill_switch_active = False
            self.kill_switch_reason = None
            self.save_state()
            try:
                from risk.kill_switch import KillSwitch

                KillSwitch.clear()
            except Exception as e:
                logger.error("Failed to clear kill switch file: %s", e)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 1.0,
    ) -> float:
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
        max_position_value = self.current_capital * (
            self.max_position_exposure_percent / 100.0
        )

        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            logger.debug(
                "Position size capped by exposure limit: %s (max %s%% of capital)",
                position_size,
                self.max_position_exposure_percent,
            )
        else:
            logger.debug(
                "Position size calculated: %s (risk: %s%%, capital: %s)",
                position_size,
                risk_percent,
                self.current_capital,
            )

        return position_size

    def get_account_risk_state(self, total_equity: Optional[float] = None):
        """Fractions aligned with risk.rules.RiskRules (e.g. 0.03 = 3%).

        When ``total_equity`` is set (mark-to-market, incl. unrealized PnL), drawdown
        and daily loss use equity ratios with values clamped to [0, 1]. When omitted,
        falls back to internal capital / realized PnL tracking for backward
        compatibility.
        """
        from risk.models import AccountRiskState

        self.reset_daily_tracking()

        if total_equity is None:
            eq = max(self.current_capital, 0.0)
            if eq <= 0:
                return AccountRiskState(0.0, 0.0)
            eq_denom = max(eq, 1e-12)
            daily_loss_fraction = max(0.0, -self.daily_pnl) / eq_denom
            daily_loss_fraction = max(0.0, min(1.0, daily_loss_fraction))
            peak = max(self.peak_capital, self.current_capital)
            max_dd = (peak - self.current_capital) / peak if peak > 0 else 0.0
            max_dd = max(0.0, min(1.0, max_dd))
            return AccountRiskState(
                daily_loss_fraction=daily_loss_fraction,
                max_drawdown_fraction=max_dd,
            )

        today_iso = datetime.now(timezone.utc).date().isoformat()
        self._hydrate_start_of_day_equity_from_file(today_iso)

        eq = max(float(total_equity), 0.0)
        if not math.isfinite(eq):
            eq = 0.0
        if eq <= 0:
            return AccountRiskState(0.0, 0.0)

        if self._start_of_day_equity is None:
            self._start_of_day_equity = eq
            self._persist_start_of_day_equity_file(today_iso, eq)

        sod = max(self._start_of_day_equity, 1e-12)
        raw_daily = (sod - eq) / sod if sod > 0 else 0.0
        daily_loss_fraction = max(0.0, min(1.0, raw_daily))

        if self._risk_peak_equity is None:
            self._risk_peak_equity = eq
        else:
            self._risk_peak_equity = max(self._risk_peak_equity, eq)

        peak_eq = max(self._risk_peak_equity, 1e-12)
        raw_dd = (peak_eq - eq) / peak_eq if peak_eq > 0 else 0.0
        max_drawdown_fraction = max(0.0, min(1.0, raw_dd))

        return AccountRiskState(
            daily_loss_fraction=daily_loss_fraction,
            max_drawdown_fraction=max_drawdown_fraction,
        )

    def _hydrate_start_of_day_equity_from_file(self, today_iso: str) -> None:
        """Restore start-of-day equity when file matches today's UTC date (restart-safe)."""
        if self._start_of_day_equity is not None:
            return
        try:
            if not os.path.isfile(DAILY_EQUITY_FILE):
                return
            with open(DAILY_EQUITY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            if data.get("date") != today_iso:
                return
            raw = data.get("start_of_day_equity")
            if raw is None:
                return
            val = float(raw)
            if not math.isfinite(val) or val < 0:
                return
            self._start_of_day_equity = val
        except Exception:
            pass

    def _persist_start_of_day_equity_file(self, today_iso: str, equity: float) -> None:
        """Write daily anchor after first live equity sample for the UTC day."""
        try:
            if not math.isfinite(equity) or equity < 0:
                return
            root = os.path.dirname(DAILY_EQUITY_FILE)
            if root:
                os.makedirs(root, exist_ok=True)
            payload = {"date": today_iso, "start_of_day_equity": float(equity)}
            with open(DAILY_EQUITY_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def sync_kill_switch_from_file(self) -> None:
        """If data/kill_switch.json is active, align in-memory halt state (VPS / external edits)."""
        try:
            from risk.kill_switch import KillSwitch

            if KillSwitch.is_active():
                if not self.kill_switch_active:
                    self.kill_switch_active = True
                    self.kill_switch_reason = KillSwitch.load().get("reason")
                    logger.critical(
                        "Kill switch active from disk: %s", self.kill_switch_reason
                    )
        except Exception:
            pass

    def get_risk_status(self) -> Dict:
        self.reset_daily_tracking()

        return {
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": self._daily_loss_limit_amount(),
            "daily_loss_limit_percent": self.daily_loss_limit_percent,
            "daily_trades_count": len(self.daily_trades),
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "within_trading_hours": self.is_within_trading_hours(),
            "can_trade": self.can_open_new_trade()[0],
            "current_date": str(self.current_date) if self.current_date else None,
        }

    def save_state(self):
        try:
            state = {
                "current_capital": self.current_capital,
                "current_date": str(self.current_date) if self.current_date else None,
                "daily_pnl": self.daily_pnl,
                "daily_trades": self.daily_trades[-10:],
                "kill_switch_active": self.kill_switch_active,
                "kill_switch_reason": self.kill_switch_reason,
                "peak_capital": self.peak_capital,
            }

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            logger.debug("Risk state saved")

        except Exception as e:
            logger.error("Error saving risk state: %s", e)

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                self.current_capital = state.get("current_capital", self.initial_capital)
                date_str = state.get("current_date")
                if date_str:
                    self.current_date = datetime.fromisoformat(date_str).date()
                self.daily_pnl = state.get("daily_pnl", 0.0)
                self.daily_trades = state.get("daily_trades", [])
                self.kill_switch_active = state.get("kill_switch_active", False)
                self.kill_switch_reason = state.get("kill_switch_reason")
                self.peak_capital = state.get("peak_capital", self.current_capital)
                self.peak_capital = max(self.peak_capital, self.current_capital)

                logger.info("Risk state loaded from file")

        except Exception as e:
            logger.warning("Could not load risk state: %s", e)

        try:
            from risk.kill_switch import KillSwitch

            if KillSwitch.is_active():
                self.kill_switch_active = True
                data = KillSwitch.load()
                self.kill_switch_reason = data.get("reason", self.kill_switch_reason)
        except Exception as e:
            logger.warning("Could not sync kill switch file: %s", e)
