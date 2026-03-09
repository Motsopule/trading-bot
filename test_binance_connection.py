"""Simple script to test a safe connection to the Binance API.

This script:
- Loads API keys from the local `.env` file
- Initializes the Binance spot client
- Calls `get_account()` to verify connectivity
- Prints a few account balances for verification

It does **not** place any orders or trades.
"""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException


def _mask_key(value: str) -> str:
    """Mask a sensitive key, showing only the first and last characters.

    Args:
        value: Original key value.

    Returns:
        Masked representation suitable for debug output.
    """
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[0]}***{value[-1]}"


def load_api_keys() -> Dict[str, str]:
    """Load Binance API credentials from the project `.env` file.

    The `.env` file is explicitly loaded from the project root based on
    the location of this script.

    Returns:
        Dictionary with `api_key` and `api_secret` keys. Any missing
        values will be set to an empty string.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_root, ".env")

    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")

    print(
        "Detected environment variables:\n"
        f"  BINANCE_API_KEY: { _mask_key(api_key) }\n"
        f"  BINANCE_SECRET_KEY: { _mask_key(api_secret) }"
    )

    return {"api_key": api_key, "api_secret": api_secret}


def validate_api_keys(api_key: str, api_secret: str) -> bool:
    """Validate that both API key and secret are present.

    Args:
        api_key: Public Binance API key.
        api_secret: Binance API secret key.

    Returns:
        True if both keys are non-empty, otherwise False.
    """
    if not api_key or not api_secret:
        print(
            "Error: Missing Binance API credentials.\n"
            "Ensure that `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` are "
            "defined in your `.env` file."
        )
        return False
    return True


def format_balances(balances: List[Dict[str, Any]], limit: int = 5) -> None:
    """Print the first few non-zero account balances.

    Args:
        balances: Raw balances list from `client.get_account()`.
        limit: Maximum number of non-zero balances to print.
    """
    non_zero = []
    for balance in balances:
        try:
            free_amount = float(balance.get("free", 0.0))
            locked_amount = float(balance.get("locked", 0.0))
        except (TypeError, ValueError):
            continue

        total = free_amount + locked_amount
        if total > 0:
            non_zero.append((balance.get("asset", "UNKNOWN"), total))

    if not non_zero:
        print("No non-zero balances found.")
        return

    print("Balances:")
    for asset, total in non_zero[:limit]:
        print(f"{asset}: {total}")


def main() -> None:
    """Run a safe connectivity check against the Binance API."""
    keys = load_api_keys()
    api_key = keys["api_key"]
    api_secret = keys["api_secret"]

    if not validate_api_keys(api_key, api_secret):
        return

    try:
        client = Client(api_key=api_key, api_secret=api_secret)

        # This call only reads account information; it does not place orders.
        account_info = client.get_account()

    except BinanceAPIException as exc:
        print("Binance API error while connecting:")
        print(exc)
        return
    except Exception as exc:
        print("Unexpected error while connecting to Binance:")
        print(exc)
        return

    print("Connected to Binance successfully")

    balances: List[Dict[str, Any]] = account_info.get("balances", [])
    format_balances(balances)


if __name__ == "__main__":
    main()

