"""Candlestick pattern recognition.

Patterns are deliberately conservative: each returns 1 for a bullish reading,
-1 for bearish, 0 for no pattern. A body or wick ratio that is ambiguous yields
0 rather than a weak guess, so strategies do not accumulate noise signals.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..model import Candle


def _body(c: Candle) -> float:
    return c.close - c.open


def _body_abs(c: Candle) -> float:
    return abs(_body(c))


def _range(c: Candle) -> float:
    return c.high - c.low


def _upper_wick(c: Candle) -> float:
    return c.high - max(c.open, c.close)


def _lower_wick(c: Candle) -> float:
    return min(c.open, c.close) - c.low


def _bullish(c: Candle) -> bool:
    return c.close >= c.open


def hammer(candles: Sequence[Candle]) -> list:
    """Small body in the upper third with a lower wick at least twice the body."""
    out: list[Optional[int]] = [None] * len(candles)
    for i, c in enumerate(candles):
        rng = _range(c)
        if rng <= 0:
            continue
        body, lw, uw = _body_abs(c), _lower_wick(c), _upper_wick(c)
        if body <= 0:
            continue
        lower_ok = lw >= 2 * body
        body_pos = (min(c.open, c.close) - c.low) / rng
        if lower_ok and body_pos <= 0.3 and uw <= body:
            out[i] = 1
    return out


def shooting_star(candles: Sequence[Candle]) -> list:
    """Hammer inverted: upper wick at least twice the body, close in lower third."""
    out: list[Optional[int]] = [None] * len(candles)
    for i, c in enumerate(candles):
        rng = _range(c)
        if rng <= 0:
            continue
        body, lw, uw = _body_abs(c), _lower_wick(c), _upper_wick(c)
        if body <= 0:
            continue
        body_pos = (c.high - max(c.open, c.close)) / rng
        if uw >= 2 * body and body_pos <= 0.3 and lw <= body:
            out[i] = -1
    return out


def bullish_engulfing(candles: Sequence[Candle]) -> list:
    """Prior red candle fully engulfed by current green candle."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        if _body(p) < 0 and _body(c) > 0:
            if c.close >= p.open and c.open <= p.close:
                out[i] = 1
    return out


def bearish_engulfing(candles: Sequence[Candle]) -> list:
    """Prior green candle fully engulfed by current red candle."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        if _body(p) > 0 and _body(c) < 0:
            if c.open >= p.close and c.close <= p.open:
                out[i] = -1
    return out


def morning_star(candles: Sequence[Candle]) -> list:
    """Three-bar reversal: long red, small indecision, long green."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2], candles[i - 1], candles[i]
        if _body(a) < 0 and _body_abs(b) < _body_abs(a) * 0.4 and _body(c) > 0:
            if c.close > (a.open + a.close) / 2:
                out[i] = 1
    return out


def evening_star(candles: Sequence[Candle]) -> list:
    """Three-bar reversal: long green, small indecision, long red."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2], candles[i - 1], candles[i]
        if _body(a) > 0 and _body_abs(b) < _body_abs(a) * 0.4 and _body(c) < 0:
            if c.close < (a.open + a.close) / 2:
                out[i] = -1
    return out


def harami(candles: Sequence[Candle]) -> list:
    """Inside bar: current body contained within the prior body. Sign = reversal."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        if _body_abs(p) <= 0:
            continue
        if max(c.open, c.close) <= max(p.open, p.close) and min(c.open, c.close) >= min(p.open, p.close):
            out[i] = -1 if _body(p) > 0 else 1
    return out


def marubozu(candles: Sequence[Candle], threshold: float = 0.9) -> list:
    """Almost no wick: a full-bodied directional bar."""
    out: list[Optional[int]] = [None] * len(candles)
    for i, c in enumerate(candles):
        rng = _range(c)
        body = _body_abs(c)
        if rng <= 0 or body / rng < threshold:
            continue
        out[i] = 1 if _body(c) > 0 else -1
    return out


def three_white_soldiers(candles: Sequence[Candle]) -> list:
    """Three consecutive higher closes with opening inside the prior body."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2], candles[i - 1], candles[i]
        if all(_bullish(x) for x in (a, b, c)):
            if a.close < b.close < c.close:
                if b.open > a.open and c.open > b.open:
                    out[i] = 1
    return out


def three_black_crows(candles: Sequence[Candle]) -> list:
    """Three consecutive lower closes with opening inside the prior body."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2], candles[i - 1], candles[i]
        if all(not _bullish(x) for x in (a, b, c)):
            if a.close > b.close > c.close:
                if b.open < a.open and c.open < b.open:
                    out[i] = -1
    return out


def inside_bar(candles: Sequence[Candle]) -> list:
    """High and low both inside the prior bar's range."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        if c.high <= p.high and c.low >= p.low:
            out[i] = 1 if _bullish(c) else -1
    return out


def outside_bar(candles: Sequence[Candle]) -> list:
    """High and low both outside the prior bar's range."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        if c.high >= p.high and c.low <= p.low:
            out[i] = 1 if _bullish(c) else -1
    return out


def tweezer_bottom(candles: Sequence[Candle], tol: float = 0.002) -> list:
    """Two bars sharing a near-identical low after a decline."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2], candles[i - 1], candles[i]
        if a.close < a.open and b.close < b.open:
            base = min(b.low, c.low)
            if base > 0 and abs(b.low - c.low) / base <= tol:
                out[i] = 1
    return out


def tweezer_top(candles: Sequence[Candle], tol: float = 0.002) -> list:
    """Two bars sharing a near-identical high after a rise."""
    out: list[Optional[int]] = [None] * len(candles)
    for i in range(2, len(candles)):
        a, b, c = candles[i - 2], candles[i - 1], candles[i]
        if a.close > a.open and b.close > b.open:
            base = max(b.high, c.high)
            if base > 0 and abs(b.high - c.high) / base <= tol:
                out[i] = -1
    return out


def all_patterns(candles: Sequence[Candle]) -> dict:
    """Every pattern, keyed by name, for the composite signal strategy."""
    return {
        "hammer": hammer(candles),
        "shooting_star": shooting_star(candles),
        "bullish_engulfing": bullish_engulfing(candles),
        "bearish_engulfing": bearish_engulfing(candles),
        "morning_star": morning_star(candles),
        "evening_star": evening_star(candles),
        "harami": harami(candles),
        "marubozu": marubozu(candles),
        "three_white_soldiers": three_white_soldiers(candles),
        "three_black_crows": three_black_crows(candles),
        "inside_bar": inside_bar(candles),
        "outside_bar": outside_bar(candles),
        "tweezer_bottom": tweezer_bottom(candles),
        "tweezer_top": tweezer_top(candles),
    }
