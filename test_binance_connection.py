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


def load_api_keys() -> Dict[str, Any]:
    """Load Binance API credentials and testnet flag from the project `.env` file.

    The `.env` file is explicitly loaded from the project root based on
    the location of this script.

    Returns:
        Dictionary with `api_key`, `api_secret`, and `testnet` keys.
        Missing credentials will be empty strings; testnet defaults to True.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_root, ".env")

    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

    api_key = (os.getenv("BINANCE_API_KEY") or "").strip()
    api_secret = (
        (os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET")) or ""
    ).strip()
    testnet_raw = (os.getenv("BINANCE_TESTNET") or "True").strip().lower()
    testnet = testnet_raw in ("true", "1", "yes")

    print(
        "Detected environment variables:\n"
        f"  BINANCE_API_KEY: {_mask_key(api_key)}\n"
        f"  BINANCE_SECRET_KEY (or BINANCE_API_SECRET): {_mask_key(api_secret)}\n"
        f"  BINANCE_TESTNET: {os.getenv('BINANCE_TESTNET', '')!r} -> testnet_mode={testnet}"
    )

    return {"api_key": api_key, "api_secret": api_secret, "testnet": testnet}


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
    testnet = keys["testnet"]

    if not validate_api_keys(api_key, api_secret):
        return

    client = None
    api_url_used = "unknown"
    try:
        if testnet:
            client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
            )
            client.API_URL = "https://testnet.binance.vision/api"
        else:
            client = Client(api_key=api_key, api_secret=api_secret)

        api_url_used = getattr(client, "API_URL", "unknown")
        print(f"Using API endpoint: {api_url_used} (testnet={testnet})")

        # This call only reads account information; it does not place orders.
        account_info = client.get_account()

    except BinanceAPIException as exc:
        print("Binance API error while connecting:")
        print(exc)
        url = getattr(client, "API_URL", api_url_used) if client else api_url_used
        print(
            "\nDiagnostics:\n"
            f"  Endpoint used: {url}\n"
            f"  Testnet mode: {testnet}\n"
            f"  API key present: {bool(api_key)} ({_mask_key(api_key)})\n"
            f"  API secret present: {bool(api_secret)} ({_mask_key(api_secret)})\n"
            "  For Spot Testnet use keys from https://testnet.binance.vision and "
            "BINANCE_TESTNET=True."
        )
        return
    except Exception as exc:
        print("Unexpected error while connecting to Binance:")
        print(exc)
        url = getattr(client, "API_URL", api_url_used) if client else api_url_used
        print(f"\nDiagnostics: endpoint={url}, testnet={testnet}")
        return

    print("Connected to Binance successfully")

    balances: List[Dict[str, Any]] = account_info.get("balances", [])
    format_balances(balances)


if __name__ == "__main__":
    main()

