MIN_TRADES = 20
WINDOW = 50


class StrategyAllocator:

    def allocate(self, strategy_name, context):
        stats = context.strategy_stats.get(strategy_name, {})

        trades = stats.get("trades", 0)
        profit_factor = stats.get("profit_factor", 1.0)

        if trades < MIN_TRADES:
            return 0.5

        if profit_factor > 1.5:
            return 1.0
        elif profit_factor > 1.2:
            return 0.6
        else:
            return 0.3
