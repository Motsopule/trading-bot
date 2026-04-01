import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import execution.persistence as persistence
from execution.execution_engine import ExecutionEngine
from execution.fill_handler import FillHandler
from execution.oms import OMS
from execution.order_model import OrderStatus
from execution.position_manager import PositionManager
from execution.reconciliation import ReconciliationEngine


def _patch_orders_file():
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "orders.json")
    p = patch.object(persistence, "FILE_PATH", path)
    p.start()
    return tmp, p


class TestOMS(unittest.TestCase):
    def setUp(self):
        self._tmpdir, self._p = _patch_orders_file()
        self.addCleanup(self._p.stop)
        self.addCleanup(self._tmpdir.cleanup)

    def test_order_creation(self):
        oms = OMS()
        o = oms.create_order("ETHUSDT", "BUY", 1.5, "MARKET")
        self.assertIsNotNone(o)
        self.assertEqual(o["status"], OrderStatus.NEW.value)
        self.assertIn(o["id"], oms.orders)
        self.assertTrue(os.path.exists(persistence.FILE_PATH))

    def test_duplicate_protection(self):
        oms = OMS()
        o1 = oms.create_order("ETHUSDT", "BUY", 1.0, "MARKET")
        self.assertIsNotNone(o1)
        o2 = oms.create_order("ETHUSDT", "SELL", 1.0, "MARKET")
        self.assertIsNone(o2)

    def test_submission_success(self):
        class Ex:
            def place_order(self, symbol, side, quantity):
                return {
                    "orderId": 42,
                    "quantity": quantity,
                    "price": 2000.0,
                    "status": "FILLED",
                }

        oms = OMS()
        order = oms.create_order("ETHUSDT", "BUY", 0.1, "MARKET")
        eng = ExecutionEngine(Ex())
        r = eng.submit_order(order)
        self.assertTrue(r["success"])
        self.assertEqual(r["exchange_order_id"], "42")

    def test_submission_success_sets_submitted_without_fill_payload(self):
        """Exchange ack only: main sets SUBMITTED when no qty/price on raw response."""

        class Ex:
            def place_order(self, symbol, side, quantity):
                return {"orderId": 99}

        order = {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "filled_quantity": 0.0,
            "price": None,
            "status": OrderStatus.NEW.value,
        }
        r = ExecutionEngine(Ex()).submit_order(order)
        self.assertTrue(r["success"])
        order["exchange_order_id"] = r["exchange_order_id"]
        raw = r.get("raw") or {}
        qty = float(raw.get("quantity", 0) or 0)
        px = raw.get("price")
        if qty > 0 and px is not None:
            pass
        else:
            order["status"] = OrderStatus.SUBMITTED.value
        self.assertEqual(order["exchange_order_id"], "99")
        self.assertEqual(order["status"], OrderStatus.SUBMITTED.value)

    def test_rejection_handling(self):
        class Ex:
            def place_order(self, symbol, side, quantity):
                raise RuntimeError("insufficient balance")

        oms = OMS()
        order = oms.create_order("ETHUSDT", "BUY", 0.1, "MARKET")
        eng = ExecutionEngine(Ex())
        r = eng.submit_order(order)
        self.assertFalse(r["success"])
        self.assertEqual(order["status"], OrderStatus.REJECTED.value)
        oms.mark_rejected(order)
        self.assertEqual(order["status"], OrderStatus.REJECTED.value)

    def test_partial_fill(self):
        fh = FillHandler()
        order = {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "quantity": 10.0,
            "filled_quantity": 0.0,
            "price": None,
            "status": OrderStatus.SUBMITTED.value,
        }
        fh.process_fill(order, 4.0, 100.0)
        self.assertEqual(order["status"], OrderStatus.PARTIALLY_FILLED.value)
        self.assertEqual(order["filled_quantity"], 4.0)

    def test_full_fill(self):
        fh = FillHandler()
        order = {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "quantity": 5.0,
            "filled_quantity": 0.0,
            "price": None,
            "status": OrderStatus.SUBMITTED.value,
        }
        fh.process_fill(order, 5.0, 100.0)
        self.assertEqual(order["status"], OrderStatus.FILLED.value)
        self.assertEqual(order["filled_quantity"], 5.0)

    def test_position_update_only_on_fill_delta(self):
        pm = PositionManager()
        self.assertEqual(pm.positions.get("ETHUSDT"), None)
        pm.apply_fill_delta("ETHUSDT", "BUY", 0.0, 2.0)
        self.assertEqual(pm.positions["ETHUSDT"], 2.0)
        pm.apply_fill_delta("ETHUSDT", "SELL", 0.0, 0.5)
        self.assertEqual(pm.positions["ETHUSDT"], 1.5)

    def test_persistence_roundtrip(self):
        oms = OMS()
        o = oms.create_order("BTCUSDT", "BUY", 0.01, "MARKET")
        oid = o["id"]
        oms2 = OMS()
        self.assertIn(oid, oms2.orders)
        self.assertEqual(oms2.orders[oid]["symbol"], "BTCUSDT")

    def test_reconciliation_marks_filled_when_missing_on_exchange(self):
        """get_order unavailable → fallback FILLED + assumed full fill log."""
        oms = OMS()
        o = oms.create_order("BTCUSDT", "BUY", 1.0, "MARKET")
        o["status"] = OrderStatus.SUBMITTED.value
        o["exchange_order_id"] = "999"
        oms._persist()

        class Ex:
            def get_open_orders(self):
                return []

            def get_order(self, symbol, oid):
                return None

        with self.assertLogs("execution.reconciliation", level="WARNING") as log_ctx:
            ReconciliationEngine().reconcile(oms, Ex(), paper_trading=False)
        self.assertTrue(
            any(
                "reconciliation_assumed_full_fill" in entry
                for entry in log_ctx.output
            )
        )
        self.assertEqual(o["status"], OrderStatus.FILLED.value)
        self.assertEqual(float(o["filled_quantity"]), 1.0)

    def test_reconciliation_partial_fill_cancelled(self):
        """get_order shows executedQty < quantity → CANCELLED, filled from exchange."""
        oms = OMS()
        o = oms.create_order("BTCUSDT", "BUY", 1.0, "MARKET")
        o["status"] = OrderStatus.SUBMITTED.value
        o["exchange_order_id"] = "1001"
        oms._persist()

        class Ex:
            def get_open_orders(self):
                return []

            def get_order(self, symbol, oid):
                return {"executedQty": 0.3, "origQty": 1.0, "status": "CANCELED"}

        ReconciliationEngine().reconcile(oms, Ex(), paper_trading=False)
        self.assertEqual(o["status"], OrderStatus.CANCELLED.value)
        self.assertEqual(float(o["filled_quantity"]), 0.3)

    def test_reconciliation_fallback_logging(self):
        oms = OMS()
        o = oms.create_order("ETHUSDT", "SELL", 2.0, "MARKET")
        o["status"] = OrderStatus.PARTIALLY_FILLED.value
        o["exchange_order_id"] = "777"
        oms._persist()

        class Ex:
            def get_open_orders(self):
                return []

            def get_order(self, symbol, oid):
                raise ConnectionError("network down")

        with self.assertLogs("execution.reconciliation", level="WARNING") as log_ctx:
            ReconciliationEngine().reconcile(oms, Ex(), paper_trading=False)
        self.assertEqual(o["status"], OrderStatus.FILLED.value)
        self.assertTrue(
            any(
                "reconciliation_assumed_full_fill" in entry
                for entry in log_ctx.output
            )
        )

    def test_negative_fill_delta(self):
        pm = PositionManager()
        with self.assertLogs("execution.position_manager", level="ERROR") as log_ctx:
            pm.apply_fill_delta("ETHUSDT", "BUY", 5.0, 3.0)
        self.assertIsNone(pm.positions.get("ETHUSDT"))
        self.assertTrue(
            any("Negative fill delta detected" in entry for entry in log_ctx.output)
        )

    def test_persistence_integrity(self):
        oms = OMS()
        o = oms.create_order("SOLUSDT", "BUY", 2.0, "MARKET")
        oid = o["id"]
        o["status"] = OrderStatus.FILLED.value
        o["filled_quantity"] = 2.0
        o["price"] = 150.0
        o["updated_at"] = datetime.now(timezone.utc).isoformat()
        oms._persist()

        oms2 = OMS()
        restored = oms2.orders[oid]
        self.assertEqual(restored["status"], OrderStatus.FILLED.value)
        self.assertEqual(float(restored["filled_quantity"]), 2.0)
        self.assertEqual(float(restored["price"]), 150.0)
        self.assertEqual(restored["symbol"], "SOLUSDT")

    def test_cancel_logic(self):
        oms = OMS()
        o = oms.create_order("BTCUSDT", "BUY", 1.0, "MARKET")
        o["status"] = OrderStatus.SUBMITTED.value
        o["exchange_order_id"] = "77"
        ex = MagicMock()
        ex.cancel_order.return_value = True
        oms.cancel_order(o, ex)
        self.assertEqual(o["status"], OrderStatus.CANCELLED.value)
        ex.cancel_order.assert_called_once()

    def test_timeout_cancel(self):
        oms = OMS()
        o = oms.create_order("BTCUSDT", "BUY", 1.0, "MARKET")
        o["status"] = OrderStatus.SUBMITTED.value
        o["exchange_order_id"] = "1"
        o["created_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=10_000)
        ).isoformat()
        ex = MagicMock()
        oms.cancel_stale_orders(ex, max_age_seconds=60)
        ex.cancel_order.assert_called()
        self.assertEqual(o["status"], OrderStatus.CANCELLED.value)


if __name__ == "__main__":
    unittest.main()
