from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from risk.models import PortfolioSnapshot, Position

# Minimum position notional (USDT) to treat as an open position (aligns with DataFetcher)
_MIN_POSITION_VALUE_USDT = 10.0


class PortfolioStateBuilder:
    """
    Builds portfolio state from the exchange (source of truth for balances/equity).

    Optional `position_enricher` supplies entry/stop/side/risk inputs from the bot's
    tracked positions; sizes are always reconciled to exchange balances when possible.
    """

    def __init__(
        self,
        exchange_client,
        symbols: Optional[List[str]] = None,
        position_enricher: Optional[Callable[[], Dict[str, dict]]] = None,
    ):
        self.exchange = exchange_client
        self.symbols = list(symbols) if symbols else []
        self.position_enricher = position_enricher

    def build(self) -> PortfolioSnapshot:
        positions = self._fetch_positions()
        equity = self._fetch_equity()
        return PortfolioSnapshot(
            equity=equity,
            positions=positions,
            timestamp=datetime.now(timezone.utc),
        )

    def _account_balances(self) -> Dict[str, float]:
        account = self.exchange.get_account()
        balances: Dict[str, float] = {}
        for b in account.get("balances", []):
            asset = b.get("asset")
            if not asset:
                continue
            total = float(b.get("free", 0) or 0) + float(b.get("locked", 0) or 0)
            if total > 0:
                balances[asset] = total
        return balances

    def _price_usdt(self, symbol: str) -> Optional[float]:
        try:
            t = self.exchange.get_symbol_ticker(symbol=symbol)
            return float(t.get("price", 0) or 0)
        except Exception:
            return None

    def _fetch_equity(self) -> float:
        balances = self._account_balances()
        usdt = balances.get("USDT", 0.0)
        total = usdt
        counted = {"USDT"}
        for asset, amt in balances.items():
            if asset in counted or amt <= 0:
                continue
            sym = f"{asset}USDT"
            px = self._price_usdt(sym)
            if px and px > 0:
                total += amt * px
        return float(total)

    def _fetch_positions(self) -> List[Position]:
        balances = self._account_balances()
        enriched: Dict[str, dict] = {}
        if self.position_enricher:
            try:
                raw = self.position_enricher()
                if isinstance(raw, dict):
                    enriched = raw
            except Exception:
                enriched = {}

        out: List[Position] = []
        symbols = self.symbols if self.symbols else list(enriched.keys())

        for symbol in symbols:
            base = symbol.replace("USDT", "")
            qty = float(balances.get(base, 0.0) or 0.0)
            mark = self._price_usdt(symbol)
            if mark is None or mark <= 0:
                continue
            notional = qty * mark
            if qty <= 0 or notional < _MIN_POSITION_VALUE_USDT:
                continue

            info = enriched.get(symbol)
            stop: Optional[float]
            if info:
                entry = float(info.get("entry_price", mark))
                raw_stop = info.get("stop_loss")
                if raw_stop is None:
                    stop = None
                else:
                    stop = float(raw_stop)
                side = str(info.get("side", "LONG"))
            else:
                entry = mark
                stop = None
                side = "LONG"

            if (
                stop is None
                or stop <= 0
                or entry <= 0
                or abs(entry - stop) <= 0
            ):
                price_risk = 0.0
            else:
                price_risk = abs(entry - stop)
            risk = qty * price_risk if price_risk > 0 else 0.0
            stop_out = float(stop) if stop is not None else 0.0

            out.append(
                Position(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    size=qty,
                    stop_loss=stop_out,
                    notional=notional,
                    risk=risk,
                )
            )

        return out
