def get_asset_class(symbol: str) -> str:
    """
    Classify asset based on symbol.

    Rules:
    - All USDT pairs → crypto
    - Future: extend for forex, indices, etc.
    """

    if not symbol:
        return "unknown"

    symbol = symbol.upper()

    # Crypto heuristic (robust and scalable)
    if symbol.endswith("USDT"):
        return "crypto"

    return "unknown"
