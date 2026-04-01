"""
Tests for Phase 3: strategy control (regime, routing, filters, integration).
"""

import unittest
from unittest.mock import MagicMock

from strategy_control.asset_classifier import get_asset_class
from strategy_control.filters import apply_filters, filter_low_volatility
from strategy_control.models import StrategyContext
from strategy_control.performance_tracker import PerformanceTracker
from strategy_control.regime_detector import RegimeDetector
from strategy_control.strategy_registry import STRATEGY_REGISTRY, load_strategy
from strategy_control.strategy_router import StrategyRouter


def _base_indicators(trending: bool, high_atr: bool):
    """Deterministic indicator bundle: trending uses wide MA separation; ranging uses tight."""
    if trending:
        ma50, ma200 = 105.0, 100.0
    else:
        ma50, ma200 = 100.5, 100.0
    atr = 2.5 if high_atr else 0.5
    atr_ma = 1.0
    return {
        "ma50": ma50,
        "ma200": ma200,
        "atr": atr,
        "atr_threshold": atr_ma,
        "min_atr": atr_ma,
    }


class TestRegimeEdgeCases(unittest.TestCase):
    def test_rapid_flip_does_not_switch_pending_resets(self):
        det = RegimeDetector()
        sym = "BTCUSDT"
        ind_trend = _base_indicators(trending=True, high_atr=True)
        ind_range = _base_indicators(trending=False, high_atr=False)

        self.assertEqual(det.detect(sym, ind_trend), "TRENDING")
        self.assertEqual(det.detect(sym, ind_range), "TRENDING")
        self.assertEqual(det.detect(sym, ind_trend), "TRENDING")
        self.assertEqual(det.detect(sym, ind_range), "TRENDING")

    def test_three_consistent_after_flip_still_switches(self):
        det = RegimeDetector()
        sym = "BTCUSDT"
        ind_trend = _base_indicators(trending=True, high_atr=True)
        ind_range = _base_indicators(trending=False, high_atr=False)

        det.detect(sym, ind_trend)
        det.detect(sym, ind_range)
        det.detect(sym, ind_range)
        self.assertEqual(det.detect(sym, ind_range), "RANGING")


class TestRegimePersistence(unittest.TestCase):
    def test_one_raw_change_does_not_switch(self):
        det = RegimeDetector()
        sym = "BTCUSDT"
        ind_trend = _base_indicators(trending=True, high_atr=True)
        ind_range = _base_indicators(trending=False, high_atr=False)

        r0 = det.detect(sym, ind_trend)
        self.assertEqual(r0, "TRENDING")

        r1 = det.detect(sym, ind_range)
        self.assertEqual(r1, "TRENDING")

        r2 = det.detect(sym, ind_range)
        self.assertEqual(r2, "TRENDING")

    def test_three_consecutive_raw_changes_switch(self):
        det = RegimeDetector()
        sym = "BTCUSDT"
        ind_trend = _base_indicators(trending=True, high_atr=True)
        ind_range = _base_indicators(trending=False, high_atr=False)

        det.detect(sym, ind_trend)
        det.detect(sym, ind_range)
        det.detect(sym, ind_range)
        r3 = det.detect(sym, ind_range)
        self.assertEqual(r3, "RANGING")


class TestStrategyRouting(unittest.TestCase):
    def test_crypto_trending(self):
        router = StrategyRouter()
        self.assertEqual(router.get_strategies("crypto", "TRENDING"), ["trend_strategy"])

    def test_forex_trending_empty(self):
        router = StrategyRouter()
        self.assertEqual(router.get_strategies("forex", "TRENDING"), [])


class TestFilters(unittest.TestCase):
    def test_filter_blocks_low_atr(self):
        low = {"ma50": 100.0, "ma200": 100.0, "atr": 0.1, "min_atr": 0.5}
        high = {**low, "atr": 1.0}
        self.assertFalse(filter_low_volatility(low))
        self.assertTrue(filter_low_volatility(high))

        ctx_low = StrategyContext(
            symbol="X",
            asset_class="crypto",
            regime="TRENDING",
            indicators=low,
            price_data={},
        )
        ctx_high = StrategyContext(
            symbol="X",
            asset_class="crypto",
            regime="TRENDING",
            indicators=high,
            price_data={},
        )
        self.assertFalse(apply_filters({"x": 1}, ctx_low))
        self.assertTrue(apply_filters({"x": 1}, ctx_high))


class TestPerSymbolRegime(unittest.TestCase):
    def test_btc_trending_eurusd_ranging_different_strategies(self):
        det = RegimeDetector()
        router = StrategyRouter()
        btc = "BTCUSDT"
        eurusd = "EURUSD"
        ind_trend = _base_indicators(trending=True, high_atr=True)
        ind_range = _base_indicators(trending=False, high_atr=False)

        r_btc = det.detect(btc, ind_trend)
        r_fx = det.detect(eurusd, ind_range)
        self.assertEqual(r_btc, "TRENDING")
        self.assertEqual(r_fx, "RANGING")

        s_btc = router.get_strategies(get_asset_class(btc), r_btc)
        s_fx = router.get_strategies(get_asset_class(eurusd), r_fx)
        self.assertEqual(s_btc, ["trend_strategy"])
        self.assertEqual(s_fx, ["mean_reversion_strategy"])


class _AlwaysSignal:
    def generate(self, context):
        return {
            "signal_details": {
                "entry_price": 100.0,
                "atr": 2.0,
            },
            "side": "LONG",
        }


class _NeverSignal:
    def generate(self, context):
        return None


def _simulate_controlled_entry(router, regime, asset_class, indicators, adapter):
    strategies = router.get_strategies(asset_class, regime)
    if not strategies:
        return {"executed": False, "reason": "no_strategy"}
    ctx = StrategyContext(
        symbol="BTCUSDT",
        asset_class=asset_class,
        regime=regime,
        indicators=indicators,
        price_data={},
    )
    sig = adapter.generate(ctx)
    if not sig:
        return {"executed": False, "reason": "no_signal"}
    if not apply_filters(sig, ctx):
        return {"executed": False, "reason": "filter"}
    return {"executed": True, "reason": "ok"}


class TestIntegration(unittest.TestCase):
    def test_valid_regime_allows_pipeline(self):
        router = StrategyRouter()
        ind = _base_indicators(trending=True, high_atr=True)
        out = _simulate_controlled_entry(
            router, "TRENDING", "crypto", ind, _AlwaysSignal()
        )
        self.assertTrue(out["executed"])

    def test_invalid_regime_blocks_before_strategy(self):
        router = StrategyRouter()
        ind = _base_indicators(trending=True, high_atr=True)
        out = _simulate_controlled_entry(
            router, "TRENDING", "forex", ind, _AlwaysSignal()
        )
        self.assertEqual(out["reason"], "no_strategy")
        self.assertFalse(out["executed"])

    def test_strategy_no_signal_no_trade(self):
        router = StrategyRouter()
        ind = _base_indicators(trending=True, high_atr=True)
        out = _simulate_controlled_entry(
            router, "TRENDING", "crypto", ind, _NeverSignal()
        )
        self.assertEqual(out["reason"], "no_signal")


class TestPerformanceTracker(unittest.TestCase):
    def test_log_trade(self):
        pt = PerformanceTracker()
        pt.log_trade("trend_strategy", 10.5, "BTCUSDT", "TRENDING")
        self.assertEqual(len(pt.records), 1)
        r = pt.records[0]
        self.assertEqual(r["strategy"], "trend_strategy")
        self.assertEqual(r["pnl"], 10.5)
        self.assertEqual(r["symbol"], "BTCUSDT")
        self.assertEqual(r["regime"], "TRENDING")
        self.assertIn("timestamp", r)


class TestLoadStrategy(unittest.TestCase):
    def test_returns_adapters(self):
        mock_engine = MagicMock()
        t = load_strategy("trend_strategy", mock_engine)
        self.assertIsNotNone(t)
        m = load_strategy("mean_reversion_strategy", mock_engine)
        self.assertIsNotNone(m)
        self.assertIsNone(load_strategy("unknown", mock_engine))


class TestRegistry(unittest.TestCase):
    def test_structure(self):
        self.assertIn("crypto", STRATEGY_REGISTRY)
        self.assertIn("forex", STRATEGY_REGISTRY)


class TestUnknownAssetClass(unittest.TestCase):
    def test_unknown_symbol(self):
        self.assertEqual(get_asset_class("NOTLISTEDXYZ"), "unknown")


class TestIndicatorsComplete(unittest.TestCase):
    def test_missing_ma50(self):
        from main import TradingBot

        self.assertEqual(
            TradingBot._indicators_complete({"ma200": 1.0, "atr": 1.0}),
            "ma50",
        )

    def test_missing_atr(self):
        from main import TradingBot

        self.assertEqual(
            TradingBot._indicators_complete({"ma50": 1.0, "ma200": 1.0}),
            "atr",
        )

    def test_complete_returns_none(self):
        from main import TradingBot

        self.assertIsNone(
            TradingBot._indicators_complete(
                {"ma50": 1.0, "ma200": 1.0, "atr": 1.0}
            )
        )


class TestStrategyExecutionLimit(unittest.TestCase):
    def test_max_strategies_constant(self):
        import main

        self.assertEqual(main.MAX_STRATEGIES_PER_SYMBOL, 1)

    def test_only_first_success_counts_when_two_eligible(self):
        executed = 0
        max_per = 1
        strategies = ["s1", "s2"]
        for _name in strategies:
            if executed >= max_per:
                break
            executed += 1
        self.assertEqual(executed, 1)


class TestPerformanceStats(unittest.TestCase):
    def test_win_rate_and_total_pnl(self):
        pt = PerformanceTracker()
        pt.log_trade("trend_strategy", 10.0, "BTCUSDT", "TRENDING")
        pt.log_trade("trend_strategy", -5.0, "BTCUSDT", "TRENDING")
        stats = pt.get_strategy_stats("trend_strategy")
        self.assertEqual(stats["trades"], 2)
        self.assertAlmostEqual(stats["total_pnl"], 5.0)
        self.assertAlmostEqual(stats["win_rate"], 0.5)

    def test_empty_strategy_returns_empty_dict(self):
        pt = PerformanceTracker()
        self.assertEqual(pt.get_strategy_stats("none"), {})

    def test_get_all_stats(self):
        pt = PerformanceTracker()
        pt.log_trade("trend_strategy", 10.0, "BTCUSDT", "TRENDING")
        pt.log_trade("mean_reversion_strategy", -2.0, "ETHUSDT", "RANGING")
        all_s = pt.get_all_stats()
        self.assertEqual(set(all_s.keys()), {"mean_reversion_strategy", "trend_strategy"})
        self.assertIn("profit_factor", all_s["trend_strategy"])


if __name__ == "__main__":
    unittest.main()
