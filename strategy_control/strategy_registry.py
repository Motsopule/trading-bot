from typing import Any, Dict, Optional

import pandas as pd

from strategy_control.models import MarketRegime


STRATEGY_REGISTRY = {
    "crypto": {
        MarketRegime.TRENDING.value: ["trend_strategy"],
    },
    "forex": {
        MarketRegime.RANGING.value: ["mean_reversion_strategy"],
    },
}


class TrendStrategyAdapter:
    """Maps regime-approved context to the existing MA trend entry logic."""

    def __init__(self, trading_engine: Any):
        self.engine = trading_engine
        self._last_no_signal_reason: Optional[str] = None

    def generate(self, context) -> Optional[Dict]:
        self._last_no_signal_reason = None
        df = context.price_data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
            return None
        data_client = context.price_data.get("data_client")
        ok, details, fail_reason = self.engine.check_entry_signal(
            df,
            symbol=context.symbol,
            data_client=data_client,
        )
        if not ok or not details:
            self._last_no_signal_reason = fail_reason
            return None
        return {"signal_details": details, "side": "LONG"}


class MeanReversionStrategyAdapter:
    """Simple ranging / mean-reversion long signal (deterministic rules)."""

    def __init__(self, trading_engine: Any):
        self.engine = trading_engine
        self._last_no_signal_reason: Optional[str] = None

    def generate(self, context) -> Optional[Dict]:
        self._last_no_signal_reason = None
        df = context.price_data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 1:
            return None
        row = df.iloc[-1]
        rsi = row.get("rsi")
        close = row.get("close")
        ma_50 = row.get("ma_50")
        atr = row.get("atr")
        if (
            rsi is None
            or pd.isna(rsi)
            or close is None
            or pd.isna(close)
            or ma_50 is None
            or pd.isna(ma_50)
            or atr is None
            or pd.isna(atr)
        ):
            return None
        if float(rsi) < 35 and float(close) < float(ma_50):
            return {
                "signal_details": {
                    "entry_price": float(close),
                    "ma_20": row.get("ma_20"),
                    "ma_50": float(ma_50),
                    "ma_200": row.get("ma_200"),
                    "atr": float(atr),
                    "side": "long",
                    "timestamp": row.name,
                },
                "side": "LONG",
            }
        self._last_no_signal_reason = (
            "rsi_not_oversold" if float(rsi) >= 35 else "close_not_below_ma50"
        )
        return None


def load_strategy(name: str, trading_engine: Any):
    if name == "trend_strategy":
        return TrendStrategyAdapter(trading_engine)
    if name == "mean_reversion_strategy":
        return MeanReversionStrategyAdapter(trading_engine)
    return None
