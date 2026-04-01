def filter_low_volatility(indicators):
    return indicators["atr"] > indicators.get("min_atr", 0)


def apply_filters(signal, context):
    if not filter_low_volatility(context.indicators):
        return False
    return True
