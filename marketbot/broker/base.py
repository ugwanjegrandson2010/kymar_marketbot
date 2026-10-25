"""Broker interface: the single seam between strategy logic and execution.

Everything a strategy does is expressed as an Order handed to a Broker. The
backtest engine, the paper broker, and the live broker all implement the same
interface, so a strategy that works in backtest works live without changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..model import Candle, Order, OrderStatus, OrderType, Position, Side, Trade


class BrokerError(Exception):
    """Raised for malformed orders or state the broker cannot execute."""


@dataclass
class Fill:
    order: Order
    price: float
    quantity: float
    fee: float
    ts: int
    slippage: float


class Broker(ABC):
    """Order placement, position tracking, and trade accounting."""

    def __init__(self, cash: float, fee_rate: float, slippage_rate: float, allow_short: bool = True):
        self.cash = float(cash)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)
        self.allow_short = allow_short
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self._open_trade: dict[str, dict] = {}

    # -- abstract ---------------------------------------------------------
    @abstractmethod
    def submit(self, order: Order) -> Order:
        """Accept, hold, or reject an order. Returns the order with status set."""
        raise NotImplementedError

    @abstractmethod
    def tick(self, candle: Candle, bar_index: int = 0) -> list[Fill]:
        """Advance to a new bar, filling any orders it can. Returns fills."""
        raise NotImplementedError

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    # -- shared accounting ------------------------------------------------
    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Cash plus mark-to-market value of open positions."""
        total = self.cash
        prices = prices or {}
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.avg_price)
            total += pos.quantity * price
        return total

    def apply_fill(self, fill: Fill, bar_index: int = 0) -> None:
        """Mutate cash and positions for a fill, and close a trade when flat."""
        order = fill.order
        sym = order.symbol
        qty = fill.quantity * order.side.sign
        cost = fill.quantity * fill.price

        self.cash -= qty * fill.price + fill.fee

        pos = self.positions.get(sym)
        if pos is None or (pos.quantity > 0) != (qty > 0):
            # Opening (or flipping) a position: record the entry.
            if pos is not None and pos.quantity != 0:
                self._close_trade(sym, fill, bar_index)
            self.positions[sym] = Position(
                symbol=sym,
                quantity=qty,
                avg_price=fill.price,
                opened_ts=fill.ts,
                side=Side.BUY if qty > 0 else Side.SELL,
            )
            self._open_trade[sym] = {
                "side": Side.BUY if qty > 0 else Side.SELL,
                "qty": abs(qty),
                "entry_price": fill.price,
                "entry_ts": fill.ts,
                "entry_bar": bar_index,
                "reason": order.reason,
            }
        else:
            pos.quantity += qty
            if abs(pos.quantity) < 1e-12:
                self.positions.pop(sym, None)
                self._close_trade(sym, fill, bar_index)
            else:
                pos.avg_price = (
                    pos.avg_price * (abs(pos.quantity) - abs(qty)) + fill.price * abs(qty)
                ) / abs(pos.quantity)

        if fill.quantity > 0:
            self.fills.append(fill)

    def _close_trade(self, symbol: str, fill: Fill, bar_index: int = 0) -> Trade | None:
        entry = self._open_trade.pop(symbol, None)
        if entry is None:
            return None
        exit_price = fill.price
        direction = entry["side"].sign
        qty = entry["qty"]
        gross = (exit_price - entry["entry_price"]) * direction * qty
        fees = sum(f.fee for f in self.fills if f.order.symbol == symbol)
        trade = Trade(
            symbol=symbol,
            side=entry["side"],
            quantity=qty,
            entry_ts=entry["entry_ts"],
            entry_price=entry["entry_price"],
            exit_ts=fill.ts,
            exit_price=exit_price,
            pnl=gross,
            pnl_pct=(gross / (entry["entry_price"] * qty)) if entry["entry_price"] * qty else 0.0,
            fees=round(fees, 8),
            bars_held=max(0, bar_index - entry.get("entry_bar", bar_index)),
            exit_reason=fill.order.reason,
            entry_reason=entry["reason"],
        )
        self.trades.append(trade)
        return trade

    def open_symbols(self) -> list[str]:
        return [s for s, p in self.positions.items() if abs(p.quantity) > 1e-12]

    def to_dict(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "trades": [t.to_dict() for t in self.trades],
            "fills": len(self.fills),
        }
