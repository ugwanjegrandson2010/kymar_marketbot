"""Core domain model: candles, signals, orders, positions, trades.

Plain dataclasses with no I/O so they can be serialized to JSON, stored in
SQLite, and compared in tests without ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class SignalAction(str, Enum):
    """What a strategy wants at a given bar."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Candle:
    ts: int                 # epoch seconds, bar open
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    symbol: str = ""        # set by providers/engine; used by the broker

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candle":
        return cls(
            ts=int(d["ts"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d.get("volume", 0.0) or 0.0),
            symbol=str(d.get("symbol", "") or ""),
        )

    @property
    def typical(self) -> float:
        """Typical price, used by several indicators."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class Signal:
    """A strategy's decision at one bar.

    strength is 0..1 and is used by the risk engine for position sizing; it is
    deliberately separate from the action so a strategy can express conviction
    without changing direction.
    """

    action: SignalAction
    symbol: str
    ts: int
    strength: float = 1.0
    reason: str = ""
    indicators: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "ts": self.ts,
            "strength": self.strength,
            "reason": self.reason,
            "indicators": self.indicators,
            "meta": self.meta,
        }


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: float
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    reason: str = ""
    ts: int = 0
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_price: Optional[float] = None
    fee: float = 0.0
    reject_reason: str = ""

    @property
    def notional(self) -> float:
        price = self.filled_price or self.limit_price or 0.0
        return self.quantity * price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "type": self.type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "reason": self.reason,
            "ts": self.ts,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "filled_price": self.filled_price,
            "fee": self.fee,
            "reject_reason": self.reject_reason,
        }


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    opened_ts: int
    side: Side = Side.BUY
    unrealized: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price + self.unrealized

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "opened_ts": self.opened_ts,
            "side": self.side.value,
            "unrealized": self.unrealized,
        }


@dataclass
class Trade:
    """A completed round trip (entry + exit), used for accuracy metrics."""

    symbol: str
    side: Side
    quantity: float
    entry_ts: int
    entry_price: float
    exit_ts: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    entry_reason: str = ""

    @property
    def won(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_ts": self.entry_ts,
            "entry_price": self.entry_price,
            "exit_ts": self.exit_ts,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 8),
            "pnl_pct": round(self.pnl_pct, 6),
            "fees": round(self.fees, 8),
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason,
            "entry_reason": self.entry_reason,
        }


def candles_from_rows(rows: list[dict]) -> list[Candle]:
    """Normalize provider output into Candle objects, dropping unusable bars."""
    out = []
    for r in rows:
        try:
            c = Candle.from_dict(r)
        except (KeyError, TypeError, ValueError):
            continue
        # A bar with a non-positive close is corrupt; skip rather than poison math.
        if not all(map(lambda x: x == x, (c.open, c.high, c.low, c.close))):
            continue  # NaN
        if c.close <= 0:
            continue
        out.append(c)
    return out
