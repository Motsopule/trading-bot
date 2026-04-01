import math

from risk.models import PortfolioSnapshot, Signal
from risk.rules import RiskRules
from risk.exposure import Exposure


class PositionSizer:

    @staticmethod
    def position_size_from_risk(
        signal: Signal,
        risk_usd: float,
        contract_multiplier: float = 1.0,
    ) -> float:
        """Convert approved USD risk to base-asset size using strict price risk."""
        if risk_usd <= 0 or not math.isfinite(float(risk_usd)):
            return 0.0
        if signal.stop_loss is None:
            return 0.0
        risk_per_unit = abs(float(signal.entry_price) - float(signal.stop_loss))
        if risk_per_unit <= 0 or not math.isfinite(risk_per_unit):
            return 0.0
        mult = float(contract_multiplier)
        if not math.isfinite(mult) or mult <= 0:
            mult = 1.0
        position_size = (float(risk_usd) / risk_per_unit) * mult
        assert position_size >= 0
        if not math.isfinite(position_size):
            return 0.0
        return float(position_size)

    @staticmethod
    def calculate(signal: Signal, portfolio: PortfolioSnapshot) -> float:
        base_risk = RiskRules.RISK_PER_TRADE * portfolio.equity

        remaining_risk = (
            RiskRules.MAX_PORTFOLIO_RISK * portfolio.equity
            - Exposure.total_risk(portfolio)
        )

        allowed_risk = min(base_risk, remaining_risk)

        # Apply slippage buffer
        allowed_risk /= RiskRules.SLIPPAGE_BUFFER

        return max(0.0, allowed_risk)
