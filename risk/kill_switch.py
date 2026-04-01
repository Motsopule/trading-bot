import json
import os
from datetime import datetime, timezone

from risk.utils import kill_switch_file_path


class KillSwitch:

    @staticmethod
    def load():
        path = kill_switch_file_path()
        if not os.path.exists(path):
            return {"active": False}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save(state):
        path = kill_switch_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)

    @staticmethod
    def trigger(reason: str):
        state = {
            "active": True,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        KillSwitch.save(state)

    @staticmethod
    def is_active():
        return bool(KillSwitch.load().get("active", False))

    @staticmethod
    def clear():
        """Deactivate persistent kill switch (manual recovery only)."""
        KillSwitch.save({"active": False})
