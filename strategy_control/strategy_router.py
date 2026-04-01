from strategy_control.strategy_registry import STRATEGY_REGISTRY


class StrategyRouter:

    def get_strategies(self, asset_class, regime):
        return STRATEGY_REGISTRY.get(asset_class, {}).get(regime, [])
