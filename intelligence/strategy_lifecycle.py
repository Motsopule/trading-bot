from intelligence.config import MIN_TRADES

_lifecycle_cache = {}


def clear_lifecycle_cache() -> None:
    """Clear status cache (e.g. tests)."""
    _lifecycle_cache.clear()


class StrategyLifecycle:

    def status(self, stats):

        trades = stats.get("trades", 0)
        pf = stats.get("profit_factor", 1.0)
        key = (trades, pf)

        if key in _lifecycle_cache:
            return _lifecycle_cache[key]

        if trades < MIN_TRADES:
            result = "WARMUP"
        elif pf < 1.0:
            result = "DISABLED"
        elif pf < 1.2:
            result = "REDUCED"
        else:
            result = "ACTIVE"

        _lifecycle_cache[key] = result
        return result
