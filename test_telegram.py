"""
Standalone script to diagnose Telegram notifications.

Usage (from project root):
  python test_telegram.py

Prints env, sends one test message via direct API, and optionally
lists chat IDs from getUpdates.
"""

import os
import sys

# Load .env from script directory
_project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_dir)
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_project_dir, ".env"))

import requests

TELEGRAM_SEND = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_GET_UPDATES = "https://api.telegram.org/bot{token}/getUpdates"


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    print("=== 1. Environment ===")
    print("TELEGRAM_BOT_TOKEN:", f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "(empty or short)")
    print("TELEGRAM_CHAT_ID:", chat_id or "(empty)")
    if not token or not chat_id:
        print("Fix: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return 1

    # Chat ID format: numeric or negative for groups
    try:
        cid = int(chat_id)
        print("Chat ID format: valid (numeric)")
    except ValueError:
        print("Chat ID format: invalid (must be numeric string, e.g. 5775323358 or -123 for group)")

    print("\n=== 2. Direct API test (sendMessage) ===")
    url = TELEGRAM_SEND.format(token=token)
    payload = {"chat_id": chat_id, "text": "Test from Trading Bot (test_telegram.py)"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        print("Status code:", resp.status_code)
        print("Response body:", resp.text)
        try:
            data = resp.json()
            if data.get("ok"):
                print("Result: OK - message delivered")
            else:
                print("Result: FAIL -", data.get("description", data))
        except Exception:
            pass
    except Exception as e:
        print("Request error:", e)

    print("\n=== 3. getUpdates (recent chats) ===")
    url_up = TELEGRAM_GET_UPDATES.format(token=token)
    try:
        r = requests.get(url_up, timeout=10)
        print("Status:", r.status_code)
        data = r.json() if r.text else {}
        if data.get("ok"):
            updates = data.get("result") or []
            if not updates:
                print("No recent updates. Start the bot with /start in Telegram to get chat_id.")
            else:
                seen = set()
                for u in updates:
                    chat = u.get("message", {}).get("chat") or u.get("callback_query", {}).get("message", {}).get("chat")
                    if chat and chat.get("id") not in seen:
                        seen.add(chat.get("id"))
                        print("  chat_id:", chat.get("id"), "type:", chat.get("type"), "title:", chat.get("title") or chat.get("username"))
        else:
            print("getUpdates error:", data.get("description", r.text))
    except Exception as e:
        print("Request error:", e)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
