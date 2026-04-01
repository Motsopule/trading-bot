import json
import os

FILE_PATH = "data/orders.json"


def load_orders():
    if not os.path.exists(FILE_PATH):
        return {}

    try:
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_orders(data):
    os.makedirs(os.path.dirname(FILE_PATH) or ".", exist_ok=True)
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, default=str)
