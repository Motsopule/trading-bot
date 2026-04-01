from typing import Dict

from risk.models import PortfolioSnapshot

from portfolio.models import OpenPositionSlice, PortfolioContext


def build_portfolio_context(
    portfolio: PortfolioSnapshot,
    strategy_by_symbol: Dict[str, str],
    strategy_stats: dict,
) -> PortfolioContext:
    """Map risk snapshot + bot strategy tags into allocation context."""
    open_positions = [
        OpenPositionSlice(
            symbol=p.symbol,
            strategy_name=strategy_by_symbol.get(p.symbol, "unknown"),
            risk=float(p.risk),
        )
        for p in portfolio.positions
    ]
    return PortfolioContext(
        equity=float(portfolio.equity),
        strategy_stats=strategy_stats,
        open_positions=open_positions,
    )
