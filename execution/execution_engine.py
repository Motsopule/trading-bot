import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Optional

from execution.order_model import OrderStatus

if TYPE_CHECKING:
    from execution.order_executor import OrderExecutor

logger = logging.getLogger(__name__)


class ExecutionEngine:

    def __init__(self, exchange_client):
        self.exchange = exchange_client

    def submit_order(self, order):

        try:
            response = self.exchange.place_order(
                symbol=order["symbol"],
                side=order["side"],
                quantity=order["quantity"],
            )

            if response is None:
                try:
                    order["status"] = OrderStatus.REJECTED.value
                    order["updated_at"] = datetime.now(timezone.utc).isoformat()
                except Exception:
                    pass
                reason = "place_order returned None"
                try:
                    logger.warning(
                        "%s",
                        json.dumps({
                            "event": "order_rejected",
                            "symbol": order.get("symbol"),
                            "reason": reason,
                        }),
                    )
                except Exception:
                    logger.warning(
                        "order_rejected symbol=%s reason=%s",
                        order.get("symbol"),
                        reason,
                    )
                return {
                    "success": False,
                    "error": reason,
                }

            oid = response.get("orderId") or response.get("order_id")
            return {
                "success": True,
                "exchange_order_id": str(oid) if oid is not None else None,
                "raw": response,
            }

        except Exception as e:
            reason = str(e)
            try:
                order["status"] = OrderStatus.REJECTED.value
                order["updated_at"] = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass
            try:
                logger.warning(
                    "%s",
                    json.dumps({
                        "event": "order_rejected",
                        "symbol": order.get("symbol"),
                        "reason": reason,
                    }),
                )
            except Exception:
                logger.warning(
                    "order_rejected symbol=%s reason=%s",
                    order.get("symbol"),
                    reason,
                )
            return {
                "success": False,
                "error": reason,
            }


class MultiSymbolExchangeAdapter:
    """
    Adapts per-symbol OrderExecutor instances to the OMS exchange interface.
    All market order placement goes through here (no bypass).
    """

    def __init__(self, executors: Dict[str, "OrderExecutor"]):
        self.executors = executors

    def place_order(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        ex = self.executors.get(symbol)
        if ex is None:
            logger.error("No executor for symbol %s", symbol)
            return None
        if side == "BUY":
            return ex.place_market_buy_order(quantity)
        if side == "SELL":
            return ex.place_market_sell_order(quantity)
        logger.error("Invalid side %s", side)
        return None

    def get_open_orders(self):
        out = []
        for sym, ex in self.executors.items():
            if ex.paper_trading:
                continue
            try:
                rows = ex.client.get_open_orders(symbol=sym)
            except Exception:
                continue
            for o in rows or []:
                out.append({"orderId": o.get("orderId"), "symbol": o.get("symbol", sym)})
        return out

    def get_order(self, symbol: str, exchange_order_id: str) -> Optional[dict]:
        ex = self.executors.get(symbol)
        if ex is None or ex.paper_trading:
            return None
        try:
            oid = int(exchange_order_id)
        except (TypeError, ValueError):
            return None
        try:
            raw = ex.client.get_order(symbol=symbol, orderId=oid)
        except Exception:
            return None
        return {
            "executedQty": raw.get("executedQty"),
            "origQty": raw.get("origQty"),
            "status": raw.get("status"),
        }

    def cancel_order(self, symbol: str, exchange_order_id: str) -> bool:
        ex = self.executors.get(symbol)
        if ex is None:
            return False
        if ex.paper_trading:
            return True
        try:
            oid = int(exchange_order_id)
        except (TypeError, ValueError):
            return False
        return ex.cancel_order(oid)
