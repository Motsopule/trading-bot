from typing import Optional

from risk.models import AccountRiskState, PortfolioSnapshot, RiskDecision, Signal
from risk.rules import RiskRules
from risk.exposure import Exposure
from risk.sizing import PositionSizer
from risk.correlation import Correlation
from risk.kill_switch import KillSwitch


class PortfolioRiskEngine:

    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        account_state: Optional[AccountRiskState] = None,
        size_override: Optional[float] = None,
    ) -> RiskDecision:

        portfolio_risk = Exposure.total_risk(portfolio)

        if KillSwitch.is_active():
            return RiskDecision(False, 0.0, "Kill switch active", portfolio_risk, 0.0)

        if portfolio.equity <= 0:
            return RiskDecision(False, 0.0, "Invalid equity", portfolio_risk, 0.0)

        # Block trading if any open position has undefined risk (no stop → no audited risk)
        for position in portfolio.positions:
            if position.stop_loss is None or position.stop_loss == 0.0:
                return RiskDecision(
                    False,
                    0.0,
                    "Open position with undefined risk",
                    portfolio_risk,
                    0.0,
                )

        if signal.stop_loss is None:
            return RiskDecision(False, 0.0, "Missing stop loss", portfolio_risk, 0.0)

        risk_per_unit_pre = abs(signal.entry_price - signal.stop_loss)
        if risk_per_unit_pre <= 0:
            return RiskDecision(
                False,
                0.0,
                "Invalid stop loss (zero risk distance)",
                portfolio_risk,
                0.0,
            )

        if account_state is not None:
            if account_state.daily_loss_fraction >= RiskRules.DAILY_LOSS_LIMIT:
                return RiskDecision(False, 0.0, "Daily loss limit", portfolio_risk, 0.0)
            if account_state.max_drawdown_fraction >= RiskRules.MAX_DRAWDOWN:
                return RiskDecision(False, 0.0, "Max drawdown", portfolio_risk, 0.0)

        if Exposure.total_positions(portfolio) >= RiskRules.MAX_POSITIONS:
            return RiskDecision(False, 0.0, "Max positions reached", portfolio_risk, 0.0)

        if portfolio_risk >= RiskRules.MAX_PORTFOLIO_RISK * portfolio.equity:
            return RiskDecision(False, 0.0, "Portfolio risk exceeded", portfolio_risk, 0.0)

        allowed_risk_usd = PositionSizer.calculate(signal, portfolio)

        if size_override is not None:
            if size_override <= 0:
                return RiskDecision(
                    False,
                    0.0,
                    "Allocation blocked (zero size)",
                    portfolio_risk,
                    0.0,
                )
            risk_per_cap = abs(float(signal.entry_price) - float(signal.stop_loss))
            cap_risk_usd = float(size_override) * risk_per_cap
            allowed_risk_usd = min(allowed_risk_usd, cap_risk_usd)

        if allowed_risk_usd <= 0:
            return RiskDecision(False, 0.0, "No risk capacity", portfolio_risk, 0.0)

        sym_risk = Exposure.symbol_risk(portfolio, signal.symbol) + allowed_risk_usd
        if sym_risk >= RiskRules.MAX_SYMBOL_RISK * portfolio.equity:
            return RiskDecision(
                False, 0.0, "Symbol risk limit", portfolio_risk, float(allowed_risk_usd)
            )

        if Correlation.breached(signal, portfolio, allowed_risk_usd):
            return RiskDecision(
                False, 0.0, "Correlation limit", portfolio_risk, float(allowed_risk_usd)
            )

        max_risk = RiskRules.MAX_PORTFOLIO_RISK * portfolio.equity
        if portfolio_risk + allowed_risk_usd > max_risk:
            return RiskDecision(
                False,
                0.0,
                "Portfolio risk exceeded (with new trade)",
                portfolio_risk,
                float(allowed_risk_usd),
            )

        qty = PositionSizer.position_size_from_risk(signal, allowed_risk_usd)
        if qty <= 0:
            return RiskDecision(False, 0.0, "Zero position size", portfolio_risk, 0.0)

        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        proposed_risk = qty * risk_per_unit
        if portfolio_risk + proposed_risk > max_risk:
            return RiskDecision(
                False,
                0.0,
                "Portfolio risk cap exceeded (final check)",
                portfolio_risk,
                float(proposed_risk),
            )

        return RiskDecision(True, qty, "Approved", portfolio_risk, float(proposed_risk))
