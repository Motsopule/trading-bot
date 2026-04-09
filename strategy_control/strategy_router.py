import json
import logging

from strategy_control.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class StrategyRouter:

    def get_strategies(self, asset_class, regime, symbol=None):
        effective_regime = "TRENDING" if regime == "HIGH_VOL" else regime
        if regime == "HIGH_VOL":
            logger.info(
                json.dumps(
                    {
                        "event": "regime_normalized",
                        "symbol": symbol,
                        "from": "HIGH_VOL",
                        "to": "TRENDING",
                    }
                )
            )
        strategies = STRATEGY_REGISTRY.get(asset_class, {}).get(
            effective_regime, []
        )
        return strategies, effective_regime

    def route(self, asset_class, regime, symbol=None):
        return self.get_strategies(asset_class, regime, symbol=symbol)
