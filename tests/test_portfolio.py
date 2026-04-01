import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from risk.engine import PortfolioRiskEngine
from risk.models import AccountRiskState, PortfolioSnapshot, Position, Signal
from risk.sizing import PositionSizer

from portfolio.allocator import PortfolioAllocator
from portfolio.correlation_engine import CORRELATED_GROUPS, group_exposure
from portfolio.exposure_tracker import ExposureTracker
from portfolio.models import OpenPositionSlice, PortfolioContext
from portfolio.portfolio_context import build_portfolio_context
from portfolio.strategy_allocator import MIN_TRADES, StrategyAllocator


def _snap(equity: float, positions) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=equity,
        positions=list(positions),
        timestamp=datetime.now(timezone.utc),
    )


class TestStrategyAllocatorWeights(unittest.TestCase):
    def test_profit_factor_high_returns_full_weight(self):
        a = StrategyAllocator()
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 2.0}},
            [],
        )
        self.assertEqual(a.allocate("trend_strategy", ctx), 1.0)

    def test_profit_factor_low_returns_reduced_weight(self):
        a = StrategyAllocator()
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 1.0}},
            [],
        )
        self.assertEqual(a.allocate("trend_strategy", ctx), 0.3)

    def test_profit_factor_mid_tier(self):
        a = StrategyAllocator()
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 1.25}},
            [],
        )
        self.assertEqual(a.allocate("trend_strategy", ctx), 0.6)


class TestMinimumTrades(unittest.TestCase):
    def test_trades_below_min_returns_half_weight(self):
        a = StrategyAllocator()
        ctx = PortfolioContext(
            10000.0,
            {
                "trend_strategy": {
                    "trades": MIN_TRADES - 1,
                    "profit_factor": 3.0,
                }
            },
            [],
        )
        self.assertEqual(a.allocate("trend_strategy", ctx), 0.5)


class TestAllocationSafety(unittest.TestCase):
    def test_adjusted_size_never_exceeds_base(self):
        alloc = PortfolioAllocator()
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 10.0}},
            [],
        )
        base = 10.0
        adj = alloc.adjust_size("trend_strategy", "BTCUSDT", base, ctx)
        self.assertLessEqual(adj, base)


class TestStrategyRiskCap(unittest.TestCase):
    def test_blocks_when_at_cap(self):
        alloc = PortfolioAllocator()
        # trend_strategy cap 3% of 10k = 300; current exposure 300
        open_positions = [
            OpenPositionSlice("BTCUSDT", "trend_strategy", 300.0),
        ]
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 2.0}},
            open_positions,
        )
        self.assertEqual(
            alloc.adjust_size("trend_strategy", "ETHUSDT", 5.0, ctx),
            0,
        )


class TestCorrelationCap(unittest.TestCase):
    def test_blocks_when_group_at_cap(self):
        alloc = PortfolioAllocator()
        sym = CORRELATED_GROUPS["crypto"][0]
        open_positions = [
            OpenPositionSlice(sym, "trend_strategy", 500.0),
        ]
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 2.0}},
            open_positions,
        )
        self.assertEqual(
            alloc.adjust_size("trend_strategy", "ETHUSDT", 5.0, ctx),
            0,
        )

    def test_group_exposure_sums_group(self):
        ctx = PortfolioContext(
            10000.0,
            {},
            [
                OpenPositionSlice("BTCUSDT", "a", 100.0),
                OpenPositionSlice("ETHUSDT", "b", 50.0),
            ],
        )
        self.assertEqual(group_exposure(ctx, "SOLUSDT"), 150.0)


class TestExposureTracker(unittest.TestCase):
    def test_strategy_exposure(self):
        et = ExposureTracker()
        ctx = PortfolioContext(
            10000.0,
            {},
            [
                OpenPositionSlice("BTCUSDT", "trend_strategy", 40.0),
                OpenPositionSlice("ETHUSDT", "mean_reversion_strategy", 60.0),
            ],
        )
        self.assertEqual(et.strategy_exposure(ctx, "trend_strategy"), 40.0)


class TestBuildPortfolioContext(unittest.TestCase):
    def test_maps_symbols_to_strategies(self):
        p = Position(
            "BTCUSDT", "LONG", 100.0, 1.0, 90.0, 100.0, 10.0
        )
        port = _snap(5000.0, [p])
        ctx = build_portfolio_context(
            port,
            {"BTCUSDT": "trend_strategy"},
            {"trend_strategy": {"trades": 20, "profit_factor": 1.5}},
        )
        self.assertEqual(ctx.open_positions[0].strategy_name, "trend_strategy")
        self.assertEqual(ctx.open_positions[0].risk, 10.0)


@patch("risk.engine.KillSwitch.is_active", return_value=False)
class TestRiskEngineSizeOverride(unittest.TestCase):
    def test_override_reduces_quantity(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        full = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0))
        self.assertTrue(full.allowed)
        capped = eng.evaluate(
            sig, port, AccountRiskState(0.0, 0.0), size_override=full.size * 0.5
        )
        self.assertTrue(capped.allowed)
        self.assertAlmostEqual(capped.size, full.size * 0.5, places=5)

    def test_zero_override_blocked(self, _ks):
        eng = PortfolioRiskEngine()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        d = eng.evaluate(sig, port, AccountRiskState(0.0, 0.0), size_override=0.0)
        self.assertFalse(d.allowed)
        self.assertIn("Allocation blocked", d.reason)


class TestIntegrationAllocationRisk(unittest.TestCase):
    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_pipeline_reduces_size_vs_unconstrained(self, _ks):
        eng = PortfolioRiskEngine()
        alloc = PortfolioAllocator()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        base_size = PositionSizer.position_size_from_risk(
            sig,
            PositionSizer.calculate(sig, port),
        )
        ctx = PortfolioContext(
            10000.0,
            {"mean_reversion_strategy": {"trades": 25, "profit_factor": 1.0}},
            [],
        )
        adjusted = alloc.adjust_size(
            "mean_reversion_strategy", "ETHUSDT", base_size, ctx
        )
        self.assertGreater(adjusted, 0)
        self.assertLess(adjusted, base_size)

        d = eng.evaluate(
            sig, port, AccountRiskState(0.0, 0.0), size_override=adjusted
        )
        self.assertTrue(d.allowed)
        self.assertAlmostEqual(d.size, adjusted, places=5)

    @patch("risk.engine.KillSwitch.is_active", return_value=False)
    def test_execution_receives_risk_validated_size(self, _ks):
        """Allocation output is fed to risk engine; OMS would receive decision.size."""
        eng = PortfolioRiskEngine()
        alloc = PortfolioAllocator()
        sig = Signal("ETHUSDT", "LONG", 2000.0, 1900.0)
        port = _snap(10000.0, [])
        base_size = PositionSizer.position_size_from_risk(
            sig,
            PositionSizer.calculate(sig, port),
        )
        ctx = PortfolioContext(
            10000.0,
            {"trend_strategy": {"trades": 25, "profit_factor": 1.4}},
            [],
        )
        adjusted = alloc.adjust_size("trend_strategy", "ETHUSDT", base_size, ctx)
        decision = eng.evaluate(
            sig, port, AccountRiskState(0.0, 0.0), size_override=adjusted
        )
        oms = MagicMock()
        oms.create_order("ETHUSDT", "BUY", decision.size, "MARKET")
        oms.create_order.assert_called_once()
        call_kw = oms.create_order.call_args[0]
        self.assertEqual(call_kw[2], decision.size)
        self.assertEqual(decision.size, adjusted)


if __name__ == "__main__":
    unittest.main()
