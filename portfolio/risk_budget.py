STRATEGY_MAX_RISK = {
    "trend_strategy": 0.03,
    "mean_reversion_strategy": 0.02,
}


def get_strategy_risk_cap(strategy):
    return STRATEGY_MAX_RISK.get(strategy, 0.02)
