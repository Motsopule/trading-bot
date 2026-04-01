from intelligence.config import VOLATILITY_HIGH_THRESHOLD, VOLATILITY_LOW_THRESHOLD


class VolatilityEngine:

    def factor(self, atr, atr_baseline):

        if atr_baseline == 0:
            return 1.0

        ratio = atr / atr_baseline

        if ratio > VOLATILITY_HIGH_THRESHOLD:
            return 0.5
        elif ratio < VOLATILITY_LOW_THRESHOLD:
            return 1.2

        return 1.0
