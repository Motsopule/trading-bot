from portfolio.strategy_allocator import StrategyAllocator
from portfolio.risk_budget import get_strategy_risk_cap
from portfolio.exposure_tracker import ExposureTracker
from portfolio.correlation_engine import group_exposure


class PortfolioAllocator:

    def __init__(self):
        self.strategy_allocator = StrategyAllocator()
        self.exposure_tracker = ExposureTracker()

    def adjust_size(self, strategy_name, symbol, base_size, context):

        weight = self.strategy_allocator.allocate(strategy_name, context)

        adjusted_size = base_size * weight

        # NEVER increase risk
        adjusted_size = min(base_size, adjusted_size)

        # Strategy risk cap
        current_strategy_risk = self.exposure_tracker.strategy_exposure(
            context, strategy_name
        )

        max_strategy_risk = get_strategy_risk_cap(strategy_name) * context.equity

        if current_strategy_risk >= max_strategy_risk:
            return 0

        # Correlation cap (simple version)
        group_risk = group_exposure(context, symbol)
        max_group_risk = 0.05 * context.equity

        if group_risk >= max_group_risk:
            return 0

        return adjusted_size
