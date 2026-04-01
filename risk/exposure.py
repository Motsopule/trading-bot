from risk.models import PortfolioSnapshot


class Exposure:

    @staticmethod
    def total_risk(portfolio: PortfolioSnapshot) -> float:
        return sum(p.risk for p in portfolio.positions)

    @staticmethod
    def symbol_risk(portfolio: PortfolioSnapshot, symbol: str) -> float:
        return sum(p.risk for p in portfolio.positions if p.symbol == symbol)

    @staticmethod
    def total_positions(portfolio: PortfolioSnapshot) -> int:
        return len(portfolio.positions)
