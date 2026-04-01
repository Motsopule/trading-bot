from typing import List

from risk.models import PortfolioSnapshot, Signal
from risk.rules import RiskRules

CORRELATION_GROUPS = {
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
}


class Correlation:

    @staticmethod
    def group_risk(portfolio: PortfolioSnapshot, symbols: List[str]) -> float:
        symset = set(symbols)
        return sum(p.risk for p in portfolio.positions if p.symbol in symset)

    @staticmethod
    def breached(
        signal: Signal,
        portfolio: PortfolioSnapshot,
        additional_risk_usd: float,
    ) -> bool:
        """Block if group risk after adding this trade would exceed cap."""
        equity = portfolio.equity
        if equity <= 0:
            return True

        for _group, symbols in CORRELATION_GROUPS.items():
            if signal.symbol not in symbols:
                continue
            risk = Correlation.group_risk(portfolio, symbols) + additional_risk_usd
            if risk >= RiskRules.MAX_GROUP_RISK * equity:
                return True

        return False
