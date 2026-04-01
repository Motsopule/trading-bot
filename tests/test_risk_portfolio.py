import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from risk.correlation import CORRELATION_GROUPS, Correlation
from risk.engine import PortfolioRiskEngine
from risk.exposure import Exposure
from risk.kill_switch import KillSwitch
from risk.models import AccountRiskState, PortfolioSnapshot, Position, Signal
from risk.rules import RiskRules
from risk.sizing import PositionSizer
from risk.utils import risk_usd_to_position_size


def _snap(
    equity: float,
    positions,
    ts=None,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=equity,
        positions=list(positions),
        timestamp=ts or datetime.now(timezone.utc),
    )


class TestExposure(unittest.TestCase):
    def test_total_and_symbol_risk(self):
        p1 = Position(
            "BTCUSDT",
            "LONG",
            100.0,
            1.0,
            90.0,
            100.0,
            10.0,
        )
        p2 = Position(
            "ETHUSDT",
            "LONG",
            2000.0,
            2.0,
            1900.0,
            4000.0,
            200.0,
        )
        port = _snap(10000.0, [p1, p2])
        self.assertAlmostEqual(Exposure.total_risk(port), 210.0)
        self.assertAlmostEqual(Exposure.symbol_risk(port, "ETHUSDT"), 200.0)
        self.assertEqual(Exposure.total_positions(port), 2)


class TestPositionSizing(unittest.TestCase):
    def test_sizing_respects_cap(self):
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        r = PositionSizer.calculate(sig, port)
        expected = (
            min(RiskRules.RISK_PER_TRADE * 10000.0, RiskRules.MAX_PORTFOLIO_RISK * 10000.0)
            / RiskRules.SLIPPAGE_BUFFER
        )
        self.assertAlmostEqual(r, expected)

    def test_risk_to_quantity(self):
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        q = risk_usd_to_position_size(sig, 100.0)
        self.assertAlmostEqual(q, 100.0 / 100.0)

    def test_position_size_zero_when_no_risk_per_unit(self):
        sig = Signal("ETHUSDT", "LONG", 2000.0, 2000.0)
        self.assertEqual(PositionSizer.position_size_from_risk(sig, 100.0), 0.0)

    def test_position_size_missing_stop(self):
        sig = Signal("ETHUSDT", "LONG", 2000.0, None)
        self.assertEqual(PositionSizer.position_size_from_risk(sig, 100.0), 0.0)

    def test_contract_multiplier(self):
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        q = PositionSizer.position_size_from_risk(sig, 100.0, contract_multiplier=2.0)
        self.assertAlmostEqual(q, 2.0)


class TestCorrelation(unittest.TestCase):
    def test_group_breach(self):
        sig = Signal("BTCUSDT", "LONG", 50000.0, 49000.0)
        crypto_syms = CORRELATION_GROUPS["crypto"]
        pos = [
            Position(
                s,
                "LONG",
                100.0,
                1.0,
                90.0,
                100.0,
                350.0,
            )
            for s in crypto_syms
        ]
        port = _snap(10000.0, pos)
        # Group risk 3 * 350 = 1050 > 4% * 10000 = 400
        extra = RiskRules.RISK_PER_TRADE * 10000.0
        self.assertTrue(Correlation.breached(sig, port, extra))


class TestKillSwitchPersistence(unittest.TestCase):
    def test_load_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "kill_switch.json")
            with patch("risk.kill_switch.kill_switch_file_path", return_value=path):
                state = KillSwitch.load()
                self.assertFalse(state.get("active", True))

                KillSwitch.trigger("test reason")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertTrue(data["active"])
                self.assertEqual(data["reason"], "test reason")
                KillSwitch.clear()
                with open(path, "r", encoding="utf-8") as f:
                    data2 = json.load(f)
                self.assertFalse(data2.get("active", True))


class TestPortfolioRiskEngine(unittest.TestCase):
    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_engine_allows_clean_book(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertTrue(d.allowed)
        self.assertGreater(d.size, 0.0)

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_daily_loss_blocks(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        d = eng.evaluate(
            sig,
            port,
            AccountRiskState(daily_loss_fraction=0.05, max_drawdown_fraction=0.0),
        )
        self.assertFalse(d.allowed)

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_missing_stop_loss_blocked(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, None)
        port = _snap(10000.0, [])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "Missing stop loss")

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_invalid_stop_zero_distance_blocked(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 2000.0)
        port = _snap(10000.0, [])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertFalse(d.allowed)
        self.assertIn("Invalid stop loss", d.reason)

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    @patch("risk.engine.PositionSizer.position_size_from_risk", return_value=1_000_000.0)
    def test_final_portfolio_risk_guard(self, _psz, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertFalse(d.allowed)
        self.assertIn("Portfolio risk cap exceeded (final check)", d.reason)

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_open_position_undefined_risk_stop_none_blocked(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("SOLUSDT", "LONG", 100.0, 90.0)
        open_pos = SimpleNamespace(stop_loss=None, risk=0.0)
        port = _snap(10000.0, [open_pos])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "Open position with undefined risk")

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_open_position_undefined_risk_stop_zero_blocked(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("SOLUSDT", "LONG", 100.0, 90.0)
        bad = Position(
            "BTCUSDT",
            "LONG",
            50000.0,
            0.01,
            0.0,
            500.0,
            0.0,
        )
        port = _snap(10000.0, [bad])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "Open position with undefined risk")


class TestDailyEquityPersistence(unittest.TestCase):
    def test_start_of_day_restored_after_restart(self):
        from risk_manager import RiskManager

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "daily_equity.json")
            today = datetime.now(timezone.utc).date().isoformat()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"date": today, "start_of_day_equity": 10000.0},
                    f,
                )
            with patch("risk_manager.DAILY_EQUITY_FILE", path):
                rm = RiskManager(initial_capital=10000.0)
                st = rm.get_account_risk_state(9000.0)
            self.assertAlmostEqual(st.daily_loss_fraction, 0.1)


if __name__ == "__main__":
    unittest.main()
