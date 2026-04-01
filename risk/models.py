from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    notional: float
    risk: float


@dataclass
class PortfolioSnapshot:
    equity: float
    positions: List[Position]
    timestamp: datetime


@dataclass
class Signal:
    symbol: str
    side: str
    entry_price: float
    stop_loss: Optional[float] = None


@dataclass
class RiskDecision:
    allowed: bool
    size: float
    reason: str
    portfolio_risk: float = 0.0
    proposed_trade_risk: float = 0.0


@dataclass
class AccountRiskState:
    """Optional daily / drawdown context (same units as RiskRules fractions)."""

    daily_loss_fraction: float  # positive number when equity lost today, e.g. 0.02 for 2%
    max_drawdown_fraction: float  # (peak - current) / peak, positive
