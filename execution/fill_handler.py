from datetime import datetime, timezone

from execution.order_model import OrderStatus


class FillHandler:

    def process_fill(self, order, fill_qty, fill_price, persist_after=None):

        if fill_qty <= 0:
            return

        prev_filled = float(order.get("filled_quantity", 0) or 0)
        new_total = prev_filled + fill_qty

        if prev_filled > 0 and order.get("price") is not None:
            old_p = float(order["price"])
            order["price"] = (old_p * prev_filled + float(fill_price) * fill_qty) / new_total
        else:
            order["price"] = float(fill_price)

        order["filled_quantity"] = new_total

        if order["filled_quantity"] < float(order["quantity"]):
            order["status"] = OrderStatus.PARTIALLY_FILLED.value
        else:
            order["status"] = OrderStatus.FILLED.value

        order["updated_at"] = datetime.now(timezone.utc).isoformat()

        if persist_after is not None:
            try:
                persist_after()
            except Exception:
                pass
