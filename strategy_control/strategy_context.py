from strategy_control.models import StrategyContext


def make_strategy_context(symbol, asset_class, regime, indicators, price_data):
    return StrategyContext(
        symbol=symbol,
        asset_class=asset_class,
        regime=regime,
        indicators=indicators,
        price_data=price_data,
    )
