CORRELATED_GROUPS = {
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "forex": ["EURUSD", "GBPUSD"],
}


def group_exposure(portfolio, symbol):
    for _name, symbols in CORRELATED_GROUPS.items():
        if symbol in symbols:
            return sum(
                p.risk for p in portfolio.open_positions
                if p.symbol in symbols
            )

    return 0
