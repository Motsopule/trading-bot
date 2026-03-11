"""
Market scanner module: scan many markets for strategy signals without executing trades.

Exchange-agnostic: uses data_fetcher.get_candles(symbol) so the system can support
Binance, Forex brokers, Deriv, or other exchanges.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

SCAN_MARKETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "TRXUSDT", "MATICUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",
    "ETCUSDT", "FILUSDT", "ICPUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "STXUSDT", "INJUSDT", "LDOUSDT", "RNDRUSDT", "SUIUSDT", "SEIUSDT",
    "AAVEUSDT", "GALAUSDT", "FTMUSDT", "THETAUSDT", "FLOWUSDT", "CHZUSDT", "CRVUSDT",
    "DYDXUSDT", "GMXUSDT", "KAVAUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "EGLDUSDT",
    "RUNEUSDT", "SUSHIUSDT", "UNIUSDT", "COMPUSDT", "MKRUSDT", "SNXUSDT", "1INCHUSDT",
    "RSRUSDT", "ZRXUSDT", "YFIUSDT", "LRCUSDT", "ENJUSDT", "HOTUSDT", "CELOUSDT",
    "DASHUSDT", "ZECUSDT", "XTZUSDT", "ZILUSDT", "XLMUSDT", "VETUSDT", "HBARUSDT",
    "QNTUSDT", "ROSEUSDT", "OCEANUSDT", "IMXUSDT", "IDUSDT", "BLURUSDT", "PEPEUSDT",
    "1000SHIBUSDT", "GMTUSDT", "LPTUSDT", "AUDIOUSDT", "HFTUSDT", "HOOKUSDT",
    "MAGICUSDT", "PYRUSDT", "MINAUSDT", "MULTIUSDT", "NEOUSDT", "WAVESUSDT",
    "LINAUSDT", "CFXUSDT", "TWTUSDT", "FXSUSDT", "FLMUSDT", "KNCUSDT", "ARKUSDT",
    "BELUSDT", "BANDUSDT", "CVCUSDT", "DENTUSDT", "DODOUSDT", "FETUSDT", "GRTUSDT",
    "ILVUSDT", "KDAUSDT", "KLAYUSDT", "MASKUSDT", "OGNUSDT", "ONEUSDT", "QTUMUSDT",
    "RIFUSDT", "RLCUSDT", "SKLUSDT", "STORJUSDT", "SXPUSDT", "TOMOUSDT", "UMAUSDT",
    "ZENUSDT", "ANKRUSDT", "BALUSDT", "BTTCUSDT", "CELRUSDT", "CTSIUSDT", "GLMUSDT",
    "JASMYUSDT", "NKNUSDT", "OAXUSDT", "PHBUSDT",
]

SIGNAL_CANDIDATES_LOG = os.path.join("logs", "signal_candidates.csv")


def _compute_strength_score(
    close: float,
    ma_50: float,
    atr: float,
    side: str,
) -> Optional[float]:
    """
    Compute strength score for a signal.
    long_score = (close - MA50) / ATR
    short_score = (MA50 - close) / ATR
    """
    if atr is None or atr <= 0:
        return None
    try:
        if side == "long":
            return (close - ma_50) / atr
        return (ma_50 - close) / atr
    except (TypeError, ZeroDivisionError):
        return None


def scan_markets(
    strategy: Any,
    data_fetcher: Any,
    timeframe: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan all SCAN_MARKETS for strategy signals without executing trades.

    Steps:
    1. Loop through all symbols
    2. Fetch candle data via data_fetcher.get_candles(symbol)
    3. Compute indicators using existing strategy logic
    4. Check for entry signals (long and short)
    5. Compute a simple strength score per signal

    Returns list of candidates sorted by score descending, e.g.:
    [
        {"symbol": "SOLUSDT", "signal": "long", "score": 2.4},
        {"symbol": "LINKUSDT", "signal": "short", "score": 1.8},
    ]

    Restores data_fetcher symbol/timeframe after scanning so trading behaviour is unchanged.
    """
    saved_symbol = getattr(data_fetcher, "symbol", None)
    saved_timeframe = getattr(data_fetcher, "timeframe", None)
    tf = timeframe or saved_timeframe or "4h"
    required_ma = getattr(strategy, "ma_200_period", 200)
    candidates: List[Dict[str, Any]] = []

    try:
        for symbol in SCAN_MARKETS:
            try:
                df = data_fetcher.get_candles(symbol, timeframe=tf, limit=500)
                if df is None or len(df) < required_ma:
                    continue
                # Use only closed candles (drop last incomplete)
                if len(df) > 1:
                    df = df.iloc[:-1]
                if len(df) < required_ma:
                    continue

                df = strategy.calculate_indicators(df)
                if df is None or len(df) < 2:
                    continue

                row = df.iloc[-1]
                close = row.get("close")
                ma_50 = row.get("ma_50")
                atr = row.get("atr")
                if close is None or ma_50 is None or atr is None:
                    continue

                # Check long entry
                long_signal, long_details = strategy.check_entry_signal(df)
                if long_signal and long_details:
                    score = _compute_strength_score(
                        float(close), float(ma_50), float(atr), "long"
                    )
                    if score is not None:
                        candidates.append({"symbol": symbol, "signal": "long", "score": score})

                # Check short entry
                short_signal, short_details = strategy.check_short_entry_signal(df)
                if short_signal and short_details:
                    score = _compute_strength_score(
                        float(close), float(ma_50), float(atr), "short"
                    )
                    if score is not None:
                        candidates.append({"symbol": symbol, "signal": "short", "score": score})

            except Exception as e:
                logger.debug("scanner symbol=%s error=%s", symbol, e)
                continue

        candidates.sort(key=lambda x: x["score"], reverse=True)
        _log_candidates(candidates)
        return candidates

    finally:
        # Restore so main loop / trading is unaffected
        if saved_symbol and saved_timeframe:
            data_fetcher.set_trading_pair(saved_symbol, saved_timeframe)
        elif saved_symbol:
            data_fetcher.set_trading_pair(saved_symbol, tf)


def _log_candidates(candidates: List[Dict[str, Any]]) -> None:
    """Append candidate signals to logs/signal_candidates.csv."""
    if not candidates:
        return
    log_dir = os.path.dirname(SIGNAL_CANDIDATES_LOG)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_exists = os.path.exists(SIGNAL_CANDIDATES_LOG)
    with open(SIGNAL_CANDIDATES_LOG, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,symbol,signal,score\n")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for c in candidates:
            f.write(f"{ts},{c['symbol']},{c['signal']},{c['score']}\n")
    logger.info(
        "scanner wrote %d candidate(s) to %s",
        len(candidates),
        SIGNAL_CANDIDATES_LOG,
    )
