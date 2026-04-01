from intelligence.config import REGIME_WEIGHTS


class RegimeAllocator:

    def weight(self, strategy, regime):
        return REGIME_WEIGHTS.get(strategy, {}).get(regime, 0.5)
