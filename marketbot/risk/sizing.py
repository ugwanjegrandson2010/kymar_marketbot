"""Risk control: position sizing and the hard limits that halt a bot.

Two principles shape this module:
  1. Size from risk, not from conviction. Position quantity is derived from how
     much equity is willing to be lost if the stop is hit, never from a gut
     feeling about how sure the signal is.
  2. Limits are hard stops, not warnings. A breached drawdown or daily loss
     limit halts trading for the session; nothing in the strategy layer can
     override it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import Config


def _round_down(qty: float, step: float = 1e-8) -> float:
    """Trim float noise so a quantity never exceeds its intended notional."""
    return round(qty / step) * step if qty > 0 else 0.0


def position_for_risk(
    equity: float,
    risk_per_trade: float,
    entry: float,
    stop: float,
    direction: str = "long",
    max_position_pct: Optional[float] = None,
) -> float:
    """Units to buy so that hitting the stop loses exactly `risk_per_trade` of equity.

    risk_amount = equity * risk_per_trade
    risk_per_unit = abs(entry - stop)
    qty = risk_amount / risk_per_unit

    A stop at the entry price makes risk_per_unit zero; that is an unusable
    setup and the function returns 0.0 rather than dividing by zero and
    producing an infinite position.
    """
    if equity <= 0 or entry <= 0:
        return 0.0
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    risk_amount = equity * risk_per_trade
    qty = risk_amount / risk_per_unit
    if max_position_pct is not None and max_position_pct > 0:
        cap = position_value_cap(equity, entry, max_position_pct)
        qty = min(qty, cap)
    return _round_down(qty)


def position_value_cap(equity: float, price: float, max_position_pct: float) -> float:
    """Largest quantity whose notional stays under max_position_pct of equity."""
    if equity <= 0 or price <= 0 or max_position_pct <= 0:
        return 0.0
    return _round_down((equity * max_position_pct) / price)


def position_for_volatility_target(
    equity: float, target_vol: float, atr: float, price: float
) -> float:
    """Units whose per-unit ATR risk matches a fraction of equity.

    Targeting a fixed volatility budget means position size falls automatically
    as volatility rises, which is the mechanism behind risk-parity sizing.
    """
    if equity <= 0 or price <= 0 or atr <= 0 or target_vol <= 0:
        return 0.0
    unit_risk = atr
    risk_budget = equity * target_vol
    return _round_down(risk_budget / unit_risk)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
    """Fractional Kelly on a win rate and average win/loss magnitudes.

    f* = (p*b - q) / b  where b = avg_win/avg_loss, q = 1 - p

    Full Kelly is optimal only with an exactly known edge; with estimated
    parameters it is far too aggressive, so the result is multiplied by
    `fraction` (quarter-Kelly by default) and clamped to [0, 1].
    """
    if avg_loss <= 0 or avg_win < 0:
        return 0.0
    if not 0.0 < win_rate < 1.0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    full = (win_rate * b - q) / b if b > 0 else 0.0
    return max(0.0, min(1.0, full * fraction))


@dataclass
class SizingDecision:
    quantity: float
    stop_price: float
    take_profit: float
    risk_amount: float
    risk_per_unit: float
    notional: float
    method: str


def size_trade(
    equity: float,
    entry: float,
    atr: float,
    config: Config,
    *,
    direction: str = "long",
    stop_mult: float = 2.0,
    reward_mult: float = 2.0,
    strength: float = 1.0,
) -> SizingDecision:
    """Full sizing pipeline: ATR stop, risk-based quantity, capped notional.

    `strength` (0..1 from a strategy) scales the risk budget only downward, so a
    low-conviction signal takes a smaller position rather than a larger one.
    """
    if atr <= 0 or entry <= 0:
        return SizingDecision(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "invalid")
    stop_distance = atr * stop_mult
    if direction == "long":
        stop = entry - stop_distance
        take_profit = entry + atr * stop_mult * reward_mult
    else:
        stop = entry + stop_distance
        take_profit = entry - atr * stop_mult * reward_mult

    risk_per_trade = config.risk_per_trade * max(0.0, min(1.0, strength))
    qty = position_for_risk(
        equity=equity,
        risk_per_trade=risk_per_trade,
        entry=entry,
        stop=stop,
        direction=direction,
        max_position_pct=config.max_position_pct,
    )
    if qty <= 0:
        return SizingDecision(0.0, stop, take_profit, 0.0, stop_distance, 0.0, "rejected")
    return SizingDecision(
        quantity=qty,
        stop_price=round(stop, 8),
        take_profit=round(take_profit, 8),
        risk_amount=round(qty * stop_distance, 2),
        risk_per_unit=round(stop_distance, 8),
        notional=round(qty * entry, 2),
        method="risk_based",
    )
