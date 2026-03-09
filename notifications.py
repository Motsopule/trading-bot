"""
Telegram notification layer for the trading bot.

Sends alerts for trade entry, exit, kill switch, and optional status updates.
Uses raw HTTPS requests. Failures are non-fatal and logged only.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_GET_UPDATES = "https://api.telegram.org/bot{token}/getUpdates"


class TelegramNotifier:
    """
    Sends trading alerts to a Telegram chat via Bot API.
    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    """

    def __init__(self):
        """Load Telegram config from environment. No-op if not configured."""
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)
        # Debug: resolved env (mask token)
        if self.enabled:
            token_mask = f"{self.bot_token[:4]}...{self.bot_token[-4:]}" if len(self.bot_token) > 8 else "(set)"
            self.logger.info(
                "Telegram notifier enabled token=%s chat_id=%s",
                token_mask, self.chat_id
            )
        else:
            self.logger.info(
                "Telegram notifier disabled: TELEGRAM_BOT_TOKEN=%s TELEGRAM_CHAT_ID=%s",
                "set" if self.bot_token else "missing",
                "set" if self.chat_id else "missing",
            )

    def get_updates(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch getUpdates from Telegram to discover chat_id (e.g. after /start).
        Returns list of updates or None on failure.
        """
        if not self.bot_token:
            return None
        url = TELEGRAM_GET_UPDATES.format(token=self.bot_token)
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json() if resp.text else {}
            if not data.get("ok"):
                self.logger.warning("getUpdates failed: %s", data.get("description", resp.text))
                return None
            return data.get("result") or []
        except requests.RequestException as e:
            self.logger.warning("getUpdates request_error=%s", e)
            return None

    def send_message(self, text: str) -> bool:
        """
        Send a plain text message to the configured Telegram chat.

        Args:
            text: Message body (plain text).

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.enabled:
            self.logger.debug("Telegram send_message skipped: notifier disabled")
            return False
        url = TELEGRAM_API_BASE.format(token=self.bot_token)
        # chat_id: numeric string or negative for groups (keep as string for API)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            body = resp.text
            try:
                data = resp.json()
            except (ValueError, TypeError):
                data = {}
            # Log full response on failure for diagnostics
            if resp.status_code != 200:
                self.logger.warning(
                    "state=NOTIFY_FAIL provider=telegram action=send_message "
                    "status=%s body=%s",
                    resp.status_code, body,
                )
                return False
            # Telegram returns 200 OK even when request fails; check ok flag
            if not data.get("ok", False):
                desc = data.get("description", body or "unknown")
                self.logger.warning(
                    "state=NOTIFY_FAIL provider=telegram action=send_message "
                    "telegram_error=%s full_body=%s",
                    desc, body,
                )
                return False
            self.logger.debug("Telegram send_message ok message_id=%s", data.get("result", {}).get("message_id"))
            return True
        except requests.RequestException as e:
            self.logger.warning(
                "state=NOTIFY_FAIL provider=telegram action=send_message "
                "request_error=%s",
                e,
                exc_info=True,
            )
            return False

    def notify_entry(self, position_details: Dict[str, Any]) -> None:
        """
        Notify that a trade was opened.

        Args:
            position_details: Dict with entry_price, quantity, stop_loss,
                and optional 'symbol' (e.g. ETHUSDT).
        """
        symbol = position_details.get("symbol", "N/A")
        price = position_details.get("entry_price")
        stop_loss = position_details.get("stop_loss")
        size = position_details.get("quantity")
        price_str = f"{price:.2f}" if price is not None else "N/A"
        sl_str = f"{stop_loss:.2f}" if stop_loss is not None else "N/A"
        size_str = f"{size:.6f}".rstrip("0").rstrip(".") if size is not None else "N/A"
        text = (
            "🚀 TRADE OPENED\n"
            f"Symbol: {symbol}\n"
            f"Price: {price_str}\n"
            f"Stop Loss: {sl_str}\n"
            f"Size: {size_str}"
        )
        try:
            if not self.send_message(text):
                self.logger.debug("state=NOTIFY_FAIL action=notify_entry")
        except Exception as e:
            self.logger.warning(
                "state=NOTIFY_FAIL action=notify_entry error=%s", e, exc_info=False
            )

    def notify_exit(self, trade_result: Dict[str, Any]) -> None:
        """
        Notify that a trade was closed.

        Args:
            trade_result: Dict with entry_price, exit_price, pnl_percent,
                and optional 'symbol'.
        """
        symbol = trade_result.get("symbol", "N/A")
        entry = trade_result.get("entry_price")
        exit_p = trade_result.get("exit_price")
        pnl_pct = trade_result.get("pnl_percent")
        entry_str = f"{entry:.2f}" if entry is not None else "N/A"
        exit_str = f"{exit_p:.2f}" if exit_p is not None else "N/A"
        if pnl_pct is not None:
            sign = "+" if pnl_pct >= 0 else ""
            pnl_str = f"{sign}{pnl_pct:.2f}%"
        else:
            pnl_str = "N/A"
        text = (
            "📉 TRADE CLOSED\n"
            f"Symbol: {symbol}\n"
            f"Entry: {entry_str}\n"
            f"Exit: {exit_str}\n"
            f"PnL: {pnl_str}"
        )
        try:
            if not self.send_message(text):
                self.logger.debug("state=NOTIFY_FAIL action=notify_exit")
        except Exception as e:
            self.logger.warning(
                "state=NOTIFY_FAIL action=notify_exit error=%s", e, exc_info=False
            )

    def notify_kill_switch(self, reason: str) -> None:
        """
        Notify that the kill switch was activated.

        Args:
            reason: Human-readable reason (e.g. "Too many API errors").
        """
        text = "🛑 KILL SWITCH ACTIVATED\n" f"Reason: {reason or 'Unknown'}"
        try:
            if not self.send_message(text):
                self.logger.debug("state=NOTIFY_FAIL action=notify_kill_switch")
        except Exception as e:
            self.logger.warning(
                "state=NOTIFY_FAIL action=notify_kill_switch error=%s",
                e,
                exc_info=False,
            )

    def notify_status(self, risk_status: Dict[str, Any]) -> None:
        """
        Optional status update (capital, daily PnL).

        Args:
            risk_status: Dict with current_capital, daily_pnl at minimum.
        """
        capital = risk_status.get("current_capital")
        daily_pnl = risk_status.get("daily_pnl")
        cap_str = f"{capital:.0f}" if capital is not None else "N/A"
        pnl_str = f"{daily_pnl:.2f}" if daily_pnl is not None else "N/A"
        text = (
            "📊 BOT STATUS\n"
            f"Capital: {cap_str}\n"
            f"Daily PnL: {pnl_str}"
        )
        try:
            if not self.send_message(text):
                self.logger.debug("state=NOTIFY_FAIL action=notify_status")
        except Exception as e:
            self.logger.warning(
                "state=NOTIFY_FAIL action=notify_status error=%s", e, exc_info=False
            )
