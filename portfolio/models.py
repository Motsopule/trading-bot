from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class OpenPositionSlice:
    """Minimal position view for allocation (strategy exposure and correlation)."""

    symbol: str
    strategy_name: str
    risk: float


@dataclass
class PortfolioContext:
    equity: float
    strategy_stats: Dict[str, Any]
    open_positions: List[OpenPositionSlice]
