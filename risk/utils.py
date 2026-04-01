import os

from risk.models import Signal
from risk.sizing import PositionSizer


def project_root() -> str:
    """Directory containing `risk/` package (repo root)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def kill_switch_file_path() -> str:
    return os.path.join(project_root(), "data", "kill_switch.json")


def risk_usd_to_position_size(
    signal: Signal,
    risk_usd: float,
    contract_multiplier: float = 1.0,
) -> float:
    """Convert approved dollar risk to base-asset quantity."""
    return PositionSizer.position_size_from_risk(
        signal, risk_usd, contract_multiplier=contract_multiplier
    )


def make_signal(
    symbol: str,
    side: str,
    entry_price: float,
    stop_loss: float,
) -> Signal:
    return Signal(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )
