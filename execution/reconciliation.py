import json
import logging
from datetime import datetime, timezone

from execution.order_model import OrderStatus
from execution.utils import OPEN_ORDER_STATUSES

logger = logging.getLogger(__name__)


class ReconciliationEngine:

    def reconcile(self, oms, exchange, paper_trading: bool = False):
        """
        Align OMS with exchange open orders. If an order is SUBMITTED/PARTIALLY_FILLED
        on our side but not open on the exchange, query get_order when available; never
        assume full fill without logging when the API cannot be consulted.
        Skips when paper_trading (no exchange open orders to compare).
        """
        if paper_trading:
            return

        open_orders = exchange.get_open_orders()
        exchange_ids = {str(o.get("orderId")) for o in open_orders if o.get("orderId") is not None}

        mutated = False
        for order in oms.orders.values():
            st = order.get("status")
            if st not in OPEN_ORDER_STATUSES or st == OrderStatus.NEW.value:
                continue

            ex_id = order.get("exchange_order_id")
            if not ex_id:
                continue

            if str(ex_id) in exchange_ids:
                continue

            details = None
            try:
                if hasattr(exchange, "get_order"):
                    details = exchange.get_order(order["symbol"], str(ex_id))
            except Exception:
                details = None

            original_qty = float(order.get("quantity", 0) or 0)

            if details is not None:
                executed_raw = float(details.get("executedQty", 0) or 0)
                executed_qty = min(executed_raw, original_qty)
                order["filled_quantity"] = executed_qty
                if executed_raw < original_qty:
                    order["status"] = OrderStatus.CANCELLED.value
                else:
                    order["status"] = OrderStatus.FILLED.value
            else:
                order["filled_quantity"] = original_qty
                order["status"] = OrderStatus.FILLED.value
                try:
                    payload = json.dumps({
                        "event": "reconciliation_assumed_full_fill",
                        "order_id": order["id"],
                    })
                    logger.warning("%s", payload)
                except Exception:
                    logger.warning(
                        "reconciliation_assumed_full_fill order_id=%s",
                        order.get("id"),
                    )

            order["updated_at"] = datetime.now(timezone.utc).isoformat()
            mutated = True

        if mutated:
            oms._persist()
