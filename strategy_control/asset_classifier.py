ASSET_CLASS = {
    "BTCUSDT": "crypto",
    "ETHUSDT": "crypto",
    "SOLUSDT": "crypto",
    "EURUSD": "forex",
    "GBPUSD": "forex",
}


def get_asset_class(symbol):
    return ASSET_CLASS.get(symbol, "unknown")
