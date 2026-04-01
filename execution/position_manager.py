import json
import logging

logger = logging.getLogger(__name__)


class PositionManager:

    def __init__(self):
        self.positions = {}

    def apply_fill_delta(self, symbol: str, side: str, previous_filled: float, new_filled: float):
        """Update positions using incremental fill implied by cumulative filled quantity."""
        delta = float(new_filled) - float(previous_filled)
        if delta < 0:
            try:
                logger.error(
                    "Negative fill delta detected %s",
                    json.dumps({"symbol": symbol, "delta": delta}),
                )
            except Exception:
                logger.error(
                    "Negative fill delta detected symbol=%s delta=%s",
                    symbol,
                    delta,
                )
            return
        if delta == 0:
            return
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        if side == "BUY":
            self.positions[symbol] += delta
        else:
            self.positions[symbol] -= delta

    def update_from_fill(self, order):
        """Apply full cumulative filled quantity (use when syncing from exchange)."""
        symbol = order["symbol"]
        side = order["side"]
        qty = float(order.get("filled_quantity", 0) or 0)
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        if side == "BUY":
            self.positions[symbol] += qty
        else:
            self.positions[symbol] -= qty
