from datetime import datetime, timezone


class PerformanceTracker:

    def __init__(self):
        self.records = []

    def log_trade(self, strategy, pnl, symbol, regime):
        self.records.append({
            "strategy": strategy,
            "pnl": pnl,
            "symbol": symbol,
            "regime": regime,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def get_strategy_stats(self, strategy):
        trades = [t for t in self.records if t["strategy"] == strategy]

        if not trades:
            return {}

        window = 50
        recent = trades[-window:]
        gross_profit = sum(t["pnl"] for t in recent if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in recent if t["pnl"] < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 1.0

        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = sum(1 for t in trades if t["pnl"] > 0) / len(trades)

        return {
            "trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def get_all_stats(self):
        strategies = sorted({t["strategy"] for t in self.records})
        return {s: self.get_strategy_stats(s) for s in strategies}
