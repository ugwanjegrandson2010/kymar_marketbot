"""Simulated broker for backtests and paper trading.

Fills are deterministic and conservative:
  * market orders fill at the bar's open (the engine never lets a signal fill on
    the same bar it was generated, so this is not look-ahead);
  * slippage is applied adversely to the trader;
  * fees are charged per side;
  * a fill that cannot be afforded is rejected rather than silently shrinking.

Paper and live differ only in where prices come from and in the credentials
check; the accounting is identical.
"""

from __future__ import annotations

from typing import Optional

from ..model import Candle, Order, OrderStatus, OrderType, Position, Side
from .base import Broker, BrokerError, Fill


class PaperBroker(Broker):
    name = "paper"

    def __init__(
        self,
        cash: float,
        fee_rate: float,
        slippage_rate: float,
        allow_short: bool = True,
    ):
        super().__init__(cash, fee_rate, slippage_rate, allow_short)
        self._last_price: dict[str, float] = {}
        self._pending: list[Order] = []

    def get_price(self, symbol: str) -> Optional[float]:
        return self._last_price.get(symbol)

    def submit(self, order: Order) -> Order:
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "non-positive quantity"
            self.orders.append(order)
            return order
        if order.side is Side.SELL and not self.allow_short:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "shorting disabled"
            self.orders.append(order)
            return order
        if order.type is OrderType.LIMIT and order.limit_price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "limit order without a limit price"
            self.orders.append(order)
            return order
        order.status = OrderStatus.PENDING
        self._pending.append(order)
        self.orders.append(order)
        return order

    def tick(self, candle: Candle, bar_index: int = 0) -> list[Fill]:
        """Fill pending orders against this bar's open, then record the bar."""
        self._last_price[candle.symbol] = candle.close
        fills: list[Fill] = []
        remaining: list[Order] = []
        ref_price = candle.open

        for order in self._pending:
            fill_price: Optional[float] = None
            if order.type is OrderType.MARKET:
                fill_price = ref_price
            elif order.type is OrderType.LIMIT:
                if order.side is Side.BUY and ref_price <= (order.limit_price or 0):
                    fill_price = ref_price
                elif order.side is Side.SELL and ref_price >= (order.limit_price or 0):
                    fill_price = ref_price
            elif order.type is OrderType.STOP:
                if order.side is Side.BUY and ref_price >= (order.stop_price or 0):
                    fill_price = ref_price
                elif order.side is Side.SELL and ref_price <= (order.stop_price or 0):
                    fill_price = ref_price

            if fill_price is None:
                remaining.append(order)
                continue

            slip = fill_price * self.slippage_rate
            executed = fill_price + slip if order.side is Side.BUY else fill_price - slip
            fee = executed * order.quantity * self.fee_rate

            cost = executed * order.quantity
            if order.side is Side.BUY and cost + fee > self.cash:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "insufficient cash"
                continue

            order.filled_qty = order.quantity
            order.filled_price = executed
            order.fee = fee
            order.status = OrderStatus.FILLED
            fill = Fill(order=order, price=executed, quantity=order.quantity, fee=fee,
                        ts=candle.ts, slippage=slip)
            self.apply_fill(fill, bar_index)
            fills.append(fill)

        self._pending = remaining
        return fills

    def market_value(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        price = self._last_price.get(symbol, pos.avg_price)
        return pos.quantity * price

    def unrealized(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        price = self._last_price.get(symbol, pos.avg_price)
        return (price - pos.avg_price) * pos.quantity
