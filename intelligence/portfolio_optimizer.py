import json
import logging

from intelligence.regime_allocator import RegimeAllocator
from intelligence.strategy_lifecycle import StrategyLifecycle
from intelligence.volatility_engine import VolatilityEngine
from intelligence.dynamic_correlation import CorrelationEngine
from intelligence.config import LIFECYCLE_WEIGHTS

_logger = logging.getLogger(__name__)


def log_event(payload: dict) -> None:
    try:
        _logger.info("phase5_adjustment %s", json.dumps(payload))
    except Exception:
        pass


class PortfolioOptimizer:

    def __init__(self):
        self.regime_allocator = RegimeAllocator()
        self.lifecycle = StrategyLifecycle()
        self.vol_engine = VolatilityEngine()
        self.corr_engine = CorrelationEngine()

    def optimize(self, base_size, context):

        strategy = context["strategy"]
        regime = context["regime"]
        stats = context["stats"]
        atr = context["atr"]
        atr_baseline = context["atr_baseline"]
        cg = context.get
        returns_matrix = cg("returns_matrix")
        symbol_idx = cg("symbol_idx", 0)

        ra = self.regime_allocator
        lc = self.lifecycle
        ve = self.vol_engine
        ce = self.corr_engine

        regime_w = ra.weight(strategy, regime)

        lifecycle_status = lc.status(stats)
        lifecycle_w = LIFECYCLE_WEIGHTS[lifecycle_status]

        if lifecycle_status == "DISABLED":
            return 0

        vol_w = ve.factor(atr, atr_baseline)

        corr_w = ce.factor(returns_matrix, symbol_idx)

        final_weight = regime_w * lifecycle_w * vol_w * corr_w

        final_size = base_size * final_weight
        final_size = min(final_size, base_size)

        if final_weight < 0.99:
            try:
                log_event(
                    {
                        "event": "phase5_adjustment",
                        "strategy": strategy,
                        "regime_weight": regime_w,
                        "lifecycle": lifecycle_status,
                        "volatility_factor": vol_w,
                        "correlation_factor": corr_w,
                        "final_weight": final_weight,
                    }
                )
            except Exception:
                pass

        return final_size
