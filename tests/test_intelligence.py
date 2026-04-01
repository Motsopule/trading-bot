import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from intelligence.config import (
    CORRELATION_REDUCTION,
    CORRELATION_THRESHOLD,
    MIN_TRADES,
    REGIME_WEIGHTS,
)
from intelligence.dynamic_correlation import CorrelationEngine
from intelligence.models import (
    CACHE_TTL_SECONDS,
    build_returns_matrix,
    clear_returns_cache,
    _compute_returns_matrix,
)
from intelligence.portfolio_optimizer import PortfolioOptimizer
from intelligence.regime_allocator import RegimeAllocator
from intelligence.strategy_lifecycle import StrategyLifecycle, clear_lifecycle_cache
from intelligence.volatility_engine import VolatilityEngine


def _price_df(n: int = 70) -> pd.DataFrame:
    return pd.DataFrame({"close": np.linspace(100.0, 120.0, n)})


class TestRegimeWeight(unittest.TestCase):
    def test_trend_strategy_trending_full(self):
        r = RegimeAllocator()
        self.assertEqual(r.weight("trend_strategy", "TRENDING"), 1.0)

    def test_trend_strategy_ranging_reduced(self):
        r = RegimeAllocator()
        self.assertEqual(r.weight("trend_strategy", "RANGING"), 0.3)

    def test_mean_reversion_ranging_full(self):
        r = RegimeAllocator()
        self.assertEqual(r.weight("mean_reversion_strategy", "RANGING"), 1.0)

    def test_mean_reversion_trending_reduced(self):
        r = RegimeAllocator()
        self.assertEqual(r.weight("mean_reversion_strategy", "TRENDING"), 0.4)

    def test_unknown_strategy_default(self):
        r = RegimeAllocator()
        self.assertEqual(r.weight("other", "TRENDING"), 0.5)

    def test_config_matches_spec(self):
        self.assertEqual(REGIME_WEIGHTS["trend_strategy"]["TRENDING"], 1.0)


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        clear_lifecycle_cache()

    def test_warmup_when_trades_below_min(self):
        lc = StrategyLifecycle()
        self.assertEqual(
            lc.status({"trades": MIN_TRADES - 1, "profit_factor": 2.0}),
            "WARMUP",
        )

    def test_disabled_when_pf_below_one(self):
        lc = StrategyLifecycle()
        self.assertEqual(
            lc.status({"trades": MIN_TRADES, "profit_factor": 0.9}),
            "DISABLED",
        )

    def test_reduced_when_pf_below_1_2(self):
        lc = StrategyLifecycle()
        self.assertEqual(
            lc.status({"trades": MIN_TRADES, "profit_factor": 1.15}),
            "REDUCED",
        )

    def test_active_when_pf_at_least_1_2(self):
        lc = StrategyLifecycle()
        self.assertEqual(
            lc.status({"trades": MIN_TRADES, "profit_factor": 1.2}),
            "ACTIVE",
        )

    def test_empty_stats_is_warmup(self):
        lc = StrategyLifecycle()
        self.assertEqual(lc.status({}), "WARMUP")

    def test_lifecycle_cache_same_stats(self):
        lc = StrategyLifecycle()
        s = {"trades": MIN_TRADES, "profit_factor": 1.5}
        self.assertEqual(lc.status(s), lc.status(s))


class TestVolatility(unittest.TestCase):
    def test_high_atr_ratio_returns_half(self):
        v = VolatilityEngine()
        self.assertEqual(v.factor(3.1, 2.0), 0.5)

    def test_low_atr_ratio_returns_boost(self):
        v = VolatilityEngine()
        self.assertEqual(v.factor(1.0, 2.0), 1.2)

    def test_mid_ratio_returns_one(self):
        v = VolatilityEngine()
        self.assertEqual(v.factor(2.0, 2.0), 1.0)

    def test_zero_baseline_returns_one(self):
        v = VolatilityEngine()
        self.assertEqual(v.factor(1.5, 0.0), 1.0)


class TestCorrelation(unittest.TestCase):
    def test_none_returns_one(self):
        c = CorrelationEngine()
        self.assertEqual(c.factor(None, 0), 1.0)

    def test_empty_returns_one(self):
        c = CorrelationEngine()
        self.assertEqual(c.factor(np.array([]), 0), 1.0)

    def test_single_symbol_row_returns_one(self):
        c = CorrelationEngine()
        rm = np.array([[0.01, -0.02, 0.0, 0.01]])
        self.assertEqual(c.factor(rm, 0), 1.0)

    def test_two_symbol_rows_skip_corrcoef(self):
        c = CorrelationEngine()
        rm = np.vstack(
            [
                np.linspace(0.001, 0.05, 50),
                np.linspace(0.001, 0.05, 50) * 1.001,
            ]
        )
        self.assertEqual(c.factor(rm, 0), 1.0)

    def test_high_correlation_reduces(self):
        c = CorrelationEngine()
        base = np.linspace(0.001, 0.05, 50)
        rm = np.vstack([base, base * 1.001, base * 1.002])
        f0 = c.factor(rm, 0)
        self.assertEqual(f0, CORRELATION_REDUCTION)

    def test_low_correlation_returns_one(self):
        c = CorrelationEngine()
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.0, size=80)
        y = rng.normal(0, 1.0, size=80)
        rm = np.vstack([x, y])
        self.assertLess(abs(float(np.corrcoef(rm)[0, 1])), CORRELATION_THRESHOLD)
        self.assertEqual(c.factor(rm, 0), 1.0)


class TestWeightCombination(unittest.TestCase):
    def setUp(self):
        clear_returns_cache()
        clear_lifecycle_cache()

    def test_multiplicative_stacking(self):
        opt = PortfolioOptimizer()
        base = 10.0
        ctx = {
            "strategy": "trend_strategy",
            "regime": "TRENDING",
            "stats": {"trades": MIN_TRADES, "profit_factor": 2.0},
            "atr": 2.0,
            "atr_baseline": 2.0,
            "returns_matrix": None,
            "symbol_idx": 0,
        }
        with patch.object(
            opt.corr_engine, "factor", return_value=0.8
        ), patch.object(
            opt.vol_engine, "factor", return_value=0.5
        ):
            out = opt.optimize(base, ctx)
        # regime 1 * lifecycle 1 * vol 0.5 * corr 0.8 = 0.4
        self.assertAlmostEqual(out, 4.0)

    def test_final_size_never_exceeds_base(self):
        opt = PortfolioOptimizer()
        base = 100.0
        ctx = {
            "strategy": "trend_strategy",
            "regime": "TRENDING",
            "stats": {"trades": MIN_TRADES, "profit_factor": 2.0},
            "atr": 1.0,
            "atr_baseline": 2.0,
            "returns_matrix": None,
            "symbol_idx": 0,
        }
        out = opt.optimize(base, ctx)
        self.assertLessEqual(out, base)

    def test_disabled_lifecycle_returns_zero(self):
        opt = PortfolioOptimizer()
        ctx = {
            "strategy": "trend_strategy",
            "regime": "TRENDING",
            "stats": {"trades": MIN_TRADES, "profit_factor": 0.5},
            "atr": 2.0,
            "atr_baseline": 2.0,
            "returns_matrix": None,
            "symbol_idx": 0,
        }
        self.assertEqual(opt.optimize(10.0, ctx), 0.0)

    def test_performance_consistency_same_inputs(self):
        opt = PortfolioOptimizer()
        ctx = {
            "strategy": "mean_reversion_strategy",
            "regime": "RANGING",
            "stats": {"trades": 30, "profit_factor": 1.5},
            "atr": 2.0,
            "atr_baseline": 2.0,
            "returns_matrix": None,
            "symbol_idx": 0,
        }
        a = opt.optimize(2.5, ctx)
        b = opt.optimize(2.5, ctx)
        self.assertEqual(a, b)


class TestReturnsCache(unittest.TestCase):
    def setUp(self):
        clear_returns_cache()

    def test_same_symbols_reuses_cached_matrix(self):
        symbols = ["A", "B"]
        price_data = {"A": _price_df(), "B": _price_df()}
        t0 = 1_000_000.0
        with patch("intelligence.models.time.time", return_value=t0):
            m1 = build_returns_matrix(symbols, price_data)
            m2 = build_returns_matrix(symbols, price_data)
        self.assertIsNotNone(m1)
        self.assertIs(m1, m2)

    def test_after_ttl_recomputes(self):
        symbols = ["A", "B"]
        price_data = {"A": _price_df(), "B": _price_df()}
        t0 = 1_000_000.0
        with patch("intelligence.models.time.time", return_value=t0):
            m1 = build_returns_matrix(symbols, price_data)
        with patch(
            "intelligence.models.time.time",
            return_value=t0 + CACHE_TTL_SECONDS + 1.0,
        ):
            m2 = build_returns_matrix(symbols, price_data)
        np.testing.assert_array_equal(m1, m2)

    def test_cached_matches_compute(self):
        symbols = ["A", "B"]
        price_data = {"A": _price_df(), "B": _price_df()}
        clear_returns_cache()
        t0 = 1_000_000.0
        with patch("intelligence.models.time.time", return_value=t0):
            cached = build_returns_matrix(symbols, price_data)
        direct = _compute_returns_matrix(symbols, price_data)
        np.testing.assert_array_equal(cached, direct)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        clear_returns_cache()
        clear_lifecycle_cache()

    def test_full_pipeline_size(self):
        opt = PortfolioOptimizer()
        base = 2.5
        t = np.linspace(0.001, 0.02, 30)
        ctx = {
            "strategy": "mean_reversion_strategy",
            "regime": "RANGING",
            "stats": {"trades": 30, "profit_factor": 1.5},
            "atr": 2.0,
            "atr_baseline": 2.0,
            "returns_matrix": np.vstack([t, t * 1.01, t * 0.99]),
            "symbol_idx": 1,
        }
        out = opt.optimize(base, ctx)
        self.assertGreater(out, 0.0)
        self.assertLessEqual(out, base)

    def test_phase5_log_when_adjustment_meaningful(self):
        opt = PortfolioOptimizer()
        ctx = {
            "strategy": "trend_strategy",
            "regime": "RANGING",
            "stats": {"trades": MIN_TRADES, "profit_factor": 2.0},
            "atr": 2.0,
            "atr_baseline": 2.0,
            "returns_matrix": None,
            "symbol_idx": 0,
        }
        with self.assertLogs("intelligence.portfolio_optimizer", level="INFO") as cm:
            opt.optimize(1.0, ctx)
        joined = " ".join(cm.output)
        self.assertIn("phase5_adjustment", joined)
        self.assertIn("regime_weight", joined)


if __name__ == "__main__":
    unittest.main()
