"""
Portfolio risk engine (research only).

This module implements a PortfolioRiskManager that enforces a maximum
total portfolio risk across multiple open positions. It is used only
in research/backtest contexts and does not affect production trading.
"""

from __future__ import annotations


class PortfolioRiskManager:
    """
    Enforces maximum portfolio risk when multiple trades are open.

    Rule: total risk across all open positions cannot exceed max_portfolio_risk
    (default 3% of capital). Per-trade risk is typically 1%; when 4+ trades
    would exceed 3% total, position sizes must be reduced or the trade skipped.

    Example:
        1 trade  -> 1% risk allowed
        2 trades -> 1% each allowed
        3 trades -> 1% each allowed
        4 trades -> must reduce size so total risk <= 3%
    """

    MAX_PORTFOLIO_RISK_PERCENT = 3.0

    def __init__(self, capital: float) -> None:
        """
        Initialize the portfolio risk manager.

        Args:
            capital: Current portfolio equity (used to compute max risk amount).
        """
        self._capital = capital
        self._current_risk_amount: float = 0.0

    def set_capital(self, capital: float) -> None:
        """
        Update the capital used for risk limits (e.g. each bar in backtest).

        Args:
            capital: Current portfolio equity.
        """
        self._capital = capital

    def can_enter_trade(self, risk_amount: float) -> bool:
        """
        Check whether adding a new trade with the given risk is allowed.

        Args:
            risk_amount: Risk in currency (e.g. equity * 0.01 for 1%).

        Returns:
            True if current_risk + risk_amount <= max_portfolio_risk (3% of capital).
        """
        if self._capital <= 0:
            return False
        max_risk = self._capital * (self.MAX_PORTFOLIO_RISK_PERCENT / 100.0)
        return (self._current_risk_amount + risk_amount) <= max_risk

    def adjust_position_size(self, base_risk: float) -> float:
        """
        Return the allowed risk (as fraction of capital) for a new trade.

        If adding full base_risk would exceed max portfolio risk, returns
        the remaining allowed risk fraction (possibly 0).

        Args:
            base_risk: Desired risk as fraction of capital (e.g. 0.01 for 1%).

        Returns:
            Allowed risk as fraction of capital (0.0 if no room).
        """
        if self._capital <= 0:
            return 0.0
        max_risk = self._capital * (self.MAX_PORTFOLIO_RISK_PERCENT / 100.0)
        remaining = max_risk - self._current_risk_amount
        if remaining <= 0:
            return 0.0
        base_risk_amount = self._capital * base_risk
        allowed_amount = min(base_risk_amount, remaining)
        return allowed_amount / self._capital if self._capital > 0 else 0.0

    def track_position_open(self, risk_amount: float) -> None:
        """
        Register risk for a newly opened position.

        Args:
            risk_amount: Risk in currency for this position.
        """
        self._current_risk_amount += risk_amount

    def track_position_close(self, risk_amount: float) -> None:
        """
        Unregister risk when a position is closed.

        Args:
            risk_amount: Risk in currency that was used for this position.
        """
        self._current_risk_amount = max(0.0, self._current_risk_amount - risk_amount)
