from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOL = "HIGH_VOL"


@dataclass
class StrategyContext:
    symbol: str
    asset_class: str
    regime: str
    indicators: dict
    price_data: dict
