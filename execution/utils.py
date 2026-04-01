"""Shared constants and helpers for the OMS."""

OPEN_ORDER_STATUSES = frozenset(
    {"NEW", "SUBMITTED", "PARTIALLY_FILLED"}
)


def normalize_side(side) -> str:
    if hasattr(side, "value"):
        return str(side.value)
    return str(side)


def normalize_order_type(order_type) -> str:
    if hasattr(order_type, "value"):
        return str(order_type.value)
    return str(order_type)
