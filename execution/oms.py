from datetime import datetime, timezone
import uuid

from execution.order_model import OrderStatus
from execution.persistence import load_orders, save_orders
from execution.utils import normalize_order_type, normalize_side, OPEN_ORDER_STATUSES


class OMS:

    def __init__(self):
        self.orders = {}
        self._load()

    def _load(self):
        self.orders = load_orders()

    def _persist(self):
        save_orders(self.orders)

    def create_order(self, symbol, side, quantity, order_type):
        side_s = normalize_side(side)
        otype_s = normalize_order_type(order_type)

        if self._has_open_order(symbol):
            return None

        order_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        order = {
            "id": order_id,
            "exchange_order_id": None,
            "symbol": symbol,
            "side": side_s,
            "quantity": float(quantity),
            "filled_quantity": 0.0,
            "price": None,
            "order_type": otype_s,
            "status": OrderStatus.NEW.value,
            "created_at": now,
            "updated_at": now,
        }

        self.orders[order_id] = order
        self._persist()

        return order

    def mark_rejected(self, order, persist: bool = True):
        order["status"] = OrderStatus.REJECTED.value
        order["updated_at"] = datetime.now(timezone.utc).isoformat()
        if persist:
            self._persist()

    def mark_submitted(self, order, exchange_order_id: str, persist: bool = True):
        order["exchange_order_id"] = exchange_order_id
        order["status"] = OrderStatus.SUBMITTED.value
        order["updated_at"] = datetime.now(timezone.utc).isoformat()
        if persist:
            self._persist()

    def cancel_order(self, order, exchange_client):
        """Cancel on exchange when possible, then mark CANCELLED locally."""
        ex_id = order.get("exchange_order_id")
        sym = order.get("symbol")
        if ex_id and sym:
            try:
                exchange_client.cancel_order(sym, str(ex_id))
            except Exception:
                pass
        order["status"] = OrderStatus.CANCELLED.value
        order["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def cancel_stale_orders(self, exchange_client, max_age_seconds: float):
        """Cancel SUBMITTED/PARTIALLY_FILLED orders older than max_age_seconds."""
        now = datetime.now(timezone.utc)
        for order in list(self.orders.values()):
            st = order.get("status")
            if st not in (OrderStatus.SUBMITTED.value, OrderStatus.PARTIALLY_FILLED.value):
                continue
            try:
                raw = order["created_at"].replace("Z", "+00:00")
                created = datetime.fromisoformat(raw)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            age = (now - created).total_seconds()
            if age >= max_age_seconds:
                self.cancel_order(order, exchange_client)

    def _has_open_order(self, symbol):
        for o in self.orders.values():
            if o["symbol"] == symbol and o["status"] in OPEN_ORDER_STATUSES:
                return True
        return False
