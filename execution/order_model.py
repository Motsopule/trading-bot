from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderStatus(Enum):
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    id: str
    exchange_order_id: str | None
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    price: float | None
    order_type: OrderType
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
