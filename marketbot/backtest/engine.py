"""Event-driven backtest engine.

Execution model, stated precisely because it is where backtests quietly lie:

  1. At bar i the strategy is called with candles[:i+1] and returns a Signal.
  2. That signal becomes an order submitted to the broker immediately.
  3. The broker fills it at bar i+1's open, with adverse slippage and fees.

Consequence: a signal can never be filled on the bar it was generated from, and
the fill price is always a price the market actually printed after the signal.
A strategy whose edge depends on same-bar execution is simply not being tested
here.

Stops and take-profits are evaluated on each bar's high/low and are checked
before the new signal is acted on, so an exit at bar i precedes an entry at i+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..broker.base import Broker
from ..broker.paper import PaperBroker
from ..config import Config
from ..model import Candle, Order, OrderType, Side, Signal, SignalAction
from ..risk.limits import RiskLimits
from ..risk.sizing import size_trade
from .metrics import Metrics


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    equity_curve: list[float] = field(default_factory=list)
    trades: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    bars: int = 0
    halted: bool = False
    halt_reason: str = ""
    config: dict = field(default_factory=dict)
    metrics: Optional[Metrics] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "bars": self.bars,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "equity_curve": [round(e, 2) for e in self.equity_curve],
            "trades": [t.to_dict() for t in self.trades],
            "orders": [o.to_dict() for o in self.orders],
        }


class BacktestEngine:
    def __init__(self, config: Optional[Config] = None, broker: Optional[Broker] = None):
        self.config = config or Config()
        self.broker = broker or PaperBroker(
            cash=self.config.starting_cash,
            fee_rate=self.config.fee_rate,
            slippage_rate=self.config.slippage_rate,
            allow_short=self.config.allow_short,
        )
        self.limits = RiskLimits(self.config)
        # Per-instance: a class-level dict would leak stops between runs.
        self._pending_stop: dict[str, tuple[float, float]] = {}

    def run(
        self,
        symbol: str,
        candles: Sequence[Candle],
        strategy,
        *,
        warmup: int = 0,
    ) -> BacktestResult:
        candles = [self._stamp(c, symbol) for c in candles]
        n = len(candles)
        if n < 2:
            raise ValueError(f"need at least 2 candles for {symbol}, got {n}")

        strategy.prepare(candles)
        self._pending_stop = {}
        self.limits = RiskLimits(self.config)
        self.limits.note_equity(self.config.starting_cash)

        equity_curve: list[float] = []
        signals: list[Signal] = []
        warmup = max(warmup, self._warmup_for(strategy))

        for i in range(n):
            bar = candles[i]

            # 1. Fill any orders submitted on the previous bar, at this open.
            self.broker.tick(bar, i)

            # 2. Evaluate stops and targets against this bar's range.
            self._manage_exits(i, candles)

            # 3. Mark equity and enforce hard limits.
            equity = self._equity(bar)
            equity_curve.append(equity)
            # `day` is the UTC day number, not the bar index: with intraday data
            # the daily-loss limit must roll at midnight, not every bar.
            day = bar.ts // 86400
            if self.limits.check(equity, day=day)["halted"]:
                # Stop generating new risk; remaining positions are flattened
                # after the loop so the equity curve ends at a clean close.
                break

            # 4. Trade only after the strategy's lookback has warmed up.
            if i < warmup or self.limits.halted:
                continue

            sig = strategy.on_bar(i, candles[: i + 1])
            if sig is None:
                continue
            signals.append(sig)
            self._execute_signal(sig, candles, i, equity)

        if self.broker.positions:
            self._flatten(candles[-1].ts)

        # Make the curve the same length as the bar count so metrics align.
        if len(equity_curve) < n:
            equity_curve.append(self._equity(candles[-1]))
        while len(equity_curve) < n:
            equity_curve.append(equity_curve[-1])

        metrics = Metrics.from_result(symbol, equity_curve, self.broker.trades)
        return BacktestResult(
            symbol=symbol,
            strategy=strategy.name,
            equity_curve=equity_curve,
            trades=self.broker.trades,
            orders=self.broker.orders,
            signals=[s.to_dict() for s in signals],
            bars=n,
            halted=self.limits.halted,
            halt_reason=self.limits.halt_reason,
            config={
                "starting_cash": self.config.starting_cash,
                "fee_rate": self.config.fee_rate,
                "slippage_rate": self.config.slippage_rate,
                "risk_per_trade": self.config.risk_per_trade,
                "warmup_bars": warmup,
            },
            metrics=metrics,
        )

    # -- internals -------------------------------------------------------
    @staticmethod
    def _stamp(c: Candle, symbol: str) -> Candle:
        return Candle(
            ts=c.ts, open=c.open, high=c.high, low=c.low, close=c.close,
            volume=c.volume, symbol=symbol,
        )

    @staticmethod
    def _warmup_for(strategy) -> int:
        """Largest lookback the strategy declares, so early bars are skipped."""
        periods = [p for _, p in strategy.indicators]
        periods += [v for v in strategy.params.values() if isinstance(v, int) and v >= 2]
        return max(periods) + 2 if periods else 30

    def _equity(self, bar: Candle) -> float:
        prices = {s: self.broker.get_price(s) or 0.0 for s in self.broker.positions}
        prices[bar.symbol] = bar.close
        return self.broker.equity(prices)

    def _execute_signal(
        self, sig: Signal, candles: Sequence[Candle], i: int, equity: float
    ) -> None:
        symbol = sig.symbol
        pos = self.broker.positions.get(symbol)
        bar = candles[i]

        if sig.action is SignalAction.CLOSE:
            if pos is not None:
                self._submit_close(symbol, sig)
            return

        if pos is not None and abs(pos.quantity) > 1e-12:
            # Already exposed: only flip on an opposite signal.
            if (sig.action is SignalAction.SELL and pos.quantity > 0) or (
                sig.action is SignalAction.BUY and pos.quantity < 0
            ):
                self._submit_close(symbol, sig)
            return

        if sig.action is SignalAction.HOLD:
            return
        if not self.limits.can_open_position(len(self.broker.positions)):
            return

        direction = "long" if sig.action is SignalAction.BUY else "short"
        atr = self._atr_at(candles, i)
        decision = size_trade(
            equity, bar.close, atr, self.config, direction=direction, strength=sig.strength
        )
        if decision.quantity <= 0:
            return

        order = Order(
            symbol=symbol,
            side=Side.BUY if direction == "long" else Side.SELL,
            quantity=decision.quantity,
            type=OrderType.MARKET,
            reason=sig.reason or sig.action.value,
            ts=bar.ts,
        )
        self.broker.submit(order)
        self._pending_stop[symbol] = (decision.stop_price, decision.take_profit)

    def _submit_close(self, symbol: str, sig: Signal) -> None:
        pos = self.broker.positions.get(symbol)
        if pos is None or abs(pos.quantity) < 1e-12:
            return
        order = Order(
            symbol=symbol,
            side=Side.SELL if pos.quantity > 0 else Side.BUY,
            quantity=abs(pos.quantity),
            type=OrderType.MARKET,
            reason=sig.reason or "close",
            ts=sig.ts,
        )
        self.broker.submit(order)
        self._pending_stop.pop(symbol, None)

    def _manage_exits(self, i: int, candles: Sequence[Candle]) -> None:
        """Exit on stop or target using the bar's high/low.

        Longs are stopped by the low and targeted by the high; shorts are the
        mirror. If a bar gaps through both, the stop is assumed hit first, which
        is the conservative reading.
        """
        for symbol, (stop, target) in list(self._pending_stop.items()):
            pos = self.broker.positions.get(symbol)
            if pos is None:
                self._pending_stop.pop(symbol, None)
                continue
            bar = candles[i]
            hit_stop = hit_target = False
            if pos.quantity > 0:
                hit_stop = bar.low <= stop
                hit_target = bar.high >= target
            else:
                hit_stop = bar.high >= stop
                hit_target = bar.low <= target
            if hit_stop or hit_target:
                reason = "stop_loss" if hit_stop else "take_profit"
                self._submit_close(
                    symbol,
                    Signal(action=SignalAction.CLOSE, symbol=symbol, ts=bar.ts, reason=reason),
                )

    def _flatten(self, ts: int) -> None:
        for symbol, pos in list(self.broker.positions.items()):
            if abs(pos.quantity) < 1e-12:
                continue
            self.broker.submit(
                Order(
                    symbol=symbol,
                    side=Side.SELL if pos.quantity > 0 else Side.BUY,
                    quantity=abs(pos.quantity),
                    type=OrderType.MARKET,
                    reason="end_of_backtest",
                    ts=ts,
                )
            )
        self._pending_stop.clear()

    @staticmethod
    def _atr_at(candles: Sequence[Candle], i: int, period: int = 14) -> float:
        from ..indicators.core import atr

        window = candles[max(0, i - period * 3) : i + 1]
        for v in reversed(atr(window, period)):
            if v:
                return float(v)
        # Fallback keeps sizing finite when ATR is unavailable.
        return float(candles[i].close) * 0.02


def run_backtest(symbol, candles, strategy, config=None) -> BacktestResult:
    """Convenience wrapper for a single symbol."""
    return BacktestEngine(config).run(symbol, candles, strategy)
