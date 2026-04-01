import json
import logging

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Per-symbol regime with hysteresis: the confirmed regime switches only after
    the raw regime differs from the confirmed regime for 3 consecutive evaluations.
    """

    def __init__(self):
        self.last_regime = {}
        self._pending_regime = {}
        self.regime_counter = {}

    def _log_regime_update(self, symbol, confirmed, pending, streak):
        try:
            payload = {
                "event": "regime_update",
                "symbol": symbol,
                "confirmed": confirmed,
                "pending": pending,
                "streak": streak,
            }
            logger.info("strategy_control %s", json.dumps(payload))
        except Exception:
            pass

    def detect_raw(self, indicators):
        ma50 = indicators["ma50"]
        ma200 = indicators["ma200"]
        atr = indicators["atr"]

        trend_strength = abs(ma50 - ma200) / ma200

        if trend_strength > 0.02:
            return "TRENDING"

        if atr > indicators.get("atr_threshold", 0):
            return "HIGH_VOL"

        return "RANGING"

    def detect(self, symbol, indicators):
        new_regime = self.detect_raw(indicators)

        if symbol not in self.last_regime:
            self.last_regime[symbol] = new_regime
            self.regime_counter[symbol] = 1
            self._pending_regime.pop(symbol, None)
            assert self.last_regime[symbol] is not None
            self._log_regime_update(symbol, new_regime, None, 1)
            return new_regime

        confirmed = self.last_regime[symbol]
        assert confirmed is not None

        if new_regime == confirmed:
            self.regime_counter[symbol] = 0
            self._pending_regime.pop(symbol, None)
            self._log_regime_update(symbol, confirmed, None, 0)
            return confirmed

        if self._pending_regime.get(symbol) != new_regime:
            self._pending_regime[symbol] = new_regime
            self.regime_counter[symbol] = 1
        else:
            self.regime_counter[symbol] += 1

        streak = self.regime_counter[symbol]
        assert streak >= 1
        pending = self._pending_regime.get(symbol)

        if streak >= 3:
            assert new_regime != confirmed
            self.last_regime[symbol] = new_regime
            self._pending_regime.pop(symbol, None)
            self.regime_counter[symbol] = 0
            assert self.last_regime[symbol] is not None
            self._log_regime_update(symbol, new_regime, None, 0)
            return new_regime

        self._log_regime_update(symbol, confirmed, pending, streak)
        return confirmed
