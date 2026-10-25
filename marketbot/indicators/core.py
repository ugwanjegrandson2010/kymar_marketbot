"""Indicator primitives and trend indicators.

Convention that matters: every function returns a list the same length as its
input, with None in warm-up positions. A strategy reading index i therefore
never sees a value that depends on bar > i. That property is what makes the
backtester's no-look-ahead guarantee meaningful, and the tests enforce it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from ..model import Candle

Number = (int, float)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def closes(candles: Sequence[Candle]) -> list[float]:
    return [c.close for c in candles]


def _pad(n: int) -> list[None]:
    return [None] * n


def _align(values: Sequence[Optional[float]], length: int) -> list[Optional[float]]:
    """Pad a short result to `length` with trailing Nones, or truncate."""
    out = list(values)
    if len(out) < length:
        out = out + [None] * (length - len(out))
    return out[:length]


def sma(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Simple moving average."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if period <= 0:
        return out
    run = 0.0
    count = 0
    for i in range(n):
        v = values[i]
        if v is None:
            out[i] = None
            continue
        run += v
        count += 1
        if i >= period:
            old = values[i - period]
            if old is None:
                count -= 1
            else:
                run -= old
        if count >= period:
            out[i] = run / period
    return out


def ema(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Exponential moving average, seeded with the first value (standard)."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if period <= 0 or n == 0:
        return out
    k = 2.0 / (period + 1)
    prev: Optional[float] = None
    for i in range(n):
        v = values[i]
        if v is None:
            out[i] = prev
            continue
        prev = v if prev is None else v * k + prev * (1 - k)
        out[i] = prev
    return out


def wma(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Weighted moving average, most recent weighted highest."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if period <= 0:
        return out
    denom = period * (period + 1) / 2
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(v is None for v in window):
            continue
        total = sum((j + 1) * float(v) for j, v in enumerate(window))
        out[i] = total / denom
    return out


def stdev(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Population standard deviation over a rolling window."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(v is None for v in window):
            continue
        mean = sum(float(v) for v in window) / period
        var = sum((float(v) - mean) ** 2 for v in window) / period
        out[i] = math.sqrt(var)
    return out


def wilder(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Wilder's smoothing (alpha = 1/period). Used by RSI, ATR, ADX, SAR."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    prev: Optional[float] = None
    for i in range(n):
        v = values[i]
        if v is None:
            continue
        if prev is None:
            prev = float(v)
        else:
            prev = prev + (float(v) - prev) / period
        out[i] = prev
    return out


def true_range(candles: Sequence[Candle]) -> list[Optional[float]]:
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        out[i] = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
    return out


def atr(candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Average True Range, Wilder smoothed."""
    return wilder(true_range(candles), period)


def macd(
    values: Sequence[Optional[float]], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """MACD line, signal line, histogram."""
    ef = ema(values, fast)
    es = ema(values, slow)
    line = [
        (a - b) if (a is not None and b is not None) else None for a, b in zip(ef, es)
    ]
    sig = ema(line, signal)
    hist = [
        (a - b) if (a is not None and b is not None) else None for a, b in zip(line, sig)
    ]
    return line, sig, hist


def adx(candles: Sequence[Candle], period: int = 14) -> tuple[list, list, list]:
    """ADX with +DI and -DI, Wilder smoothed. Returns (adx, plus_di, minus_di)."""
    n = len(candles)
    if n < period + 1:
        return [None] * n, [None] * n, [None] * n
    tr: list[float] = [0.0]
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        up = c.high - p.high
        down = p.low - c.low
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))

    atr_w = wilder(tr, period)
    pd_w = wilder(plus_dm, period)
    md_w = wilder(minus_dm, period)

    plus_di: list[Optional[float]] = [None] * n
    minus_di: list[Optional[float]] = [None] * n
    dx: list[Optional[float]] = [None] * n
    for i in range(n):
        a = atr_w[i]
        if a is None or a == 0:
            continue
        pdi = 100.0 * (pd_w[i] or 0.0) / a
        mdi = 100.0 * (md_w[i] or 0.0) / a
        plus_di[i] = pdi
        minus_di[i] = mdi
        denom = pdi + mdi
        dx[i] = (100.0 * abs(pdi - mdi) / denom) if denom else 0.0
    return wilder(dx, period), plus_di, minus_di


def aroon(candles: Sequence[Candle], period: int = 25) -> tuple[list, list, list]:
    """Aroon up, down, and oscillator."""
    n = len(candles)
    up: list[Optional[float]] = [None] * n
    down: list[Optional[float]] = [None] * n
    for i in range(period, n):
        window = candles[i - period + 1 : i + 1]
        highest = max(c.high for c in window)
        lowest = min(c.low for c in window)
        bars_since_high = period - 1 - [c.high for c in window].index(highest)
        bars_since_low = period - 1 - [c.low for c in window].index(lowest)
        up[i] = 100.0 * (period - bars_since_high) / period
        down[i] = 100.0 * (period - bars_since_low) / period
    osc = [
        (a - b) if (a is not None and b is not None) else None for a, b in zip(up, down)
    ]
    return up, down, osc


def parabolic_sar(
    candles: Sequence[Candle], af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
) -> list[Optional[float]]:
    """Parabolic SAR. Long/short state is tracked per bar; first bar is None."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    if n < 3:
        return out
    # Seed with the first bar's direction from the initial two-bar move.
    rising = candles[1].close > candles[0].close
    af = af_start
    if rising:
        sar = candles[0].low
        ep = candles[0].high
    else:
        sar = candles[0].high
        ep = candles[0].low
    out[1] = sar
    for i in range(2, n):
        c = candles[i]
        sar = sar + af * (ep - sar)
        if rising:
            sar = min(sar, candles[i - 1].low, candles[i - 2].low)
            if c.low < sar:
                rising = False
                sar = ep
                ep = c.low
                af = af_start
            else:
                if c.high > ep:
                    ep = c.high
                    af = min(af + af_step, af_max)
        else:
            sar = max(sar, candles[i - 1].high, candles[i - 2].high)
            if c.high > sar:
                rising = True
                sar = ep
                ep = c.high
                af = af_start
            else:
                if c.low < ep:
                    ep = c.low
                    af = min(af + af_step, af_max)
        out[i] = sar
    return out


def supertrend(candles: Sequence[Candle], period: int = 10, multiplier: float = 3.0) -> list[Optional[float]]:
    """Supertrend: ATR-banded trailing stop that flips with price."""
    n = len(candles)
    a = atr(candles, period)
    out: list[Optional[float]] = [None] * n
    if n == 0:
        return out
    basic_upper: list[Optional[float]] = [None] * n
    basic_lower: list[Optional[float]] = [None] * n
    for i, c in enumerate(candles):
        if a[i] is None:
            continue
        mid = (c.high + c.low) / 2.0
        basic_upper[i] = mid + multiplier * a[i]
        basic_lower[i] = mid - multiplier * a[i]
    final_upper: list[Optional[float]] = [None] * n
    final_lower: list[Optional[float]] = [None] * n
    prev_up: Optional[float] = None
    prev_low: Optional[float] = None
    for i in range(n):
        bu, bl = basic_upper[i], basic_lower[i]
        if bu is None or bl is None:
            continue
        if prev_up is None or (candles[i - 1].close if i else bu) > prev_up:
            final_upper[i] = bu
        else:
            final_upper[i] = min(bu, prev_up)
        if prev_low is None or (candles[i - 1].close if i else bl) < prev_low:
            final_lower[i] = bl
        else:
            final_lower[i] = max(bl, prev_low)
        prev_up, prev_low = final_upper[i], final_lower[i]

    trend_up: Optional[bool] = None
    for i in range(n):
        if final_upper[i] is None or final_lower[i] is None:
            continue
        price = candles[i].close
        if trend_up is None:
            trend_up = price > final_lower[i]
            out[i] = final_lower[i] if trend_up else final_upper[i]
            continue
        if trend_up and price < final_lower[i]:
            trend_up = False
        elif (not trend_up) and price > final_upper[i]:
            trend_up = True
        out[i] = final_lower[i] if trend_up else final_upper[i]
    return out


def ichimoku(candles: Sequence[Candle], tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict:
    """Ichimoku components. Senkou spans are projected forward by kijun bars."""
    n = len(candles)
    t: list[Optional[float]] = [None] * n
    k: list[Optional[float]] = [None] * n
    span_a: list[Optional[float]] = [None] * n
    span_b: list[Optional[float]] = [None] * n
    chikou: list[Optional[float]] = [None] * n

    def midpoint(period: int) -> list[Optional[float]]:
        res: list[Optional[float]] = [None] * n
        for i in range(period - 1, n):
            w = candles[i - period + 1 : i + 1]
            res[i] = (max(c.high for c in w) + min(c.low for c in w)) / 2.0
        return res

    t = midpoint(tenkan)
    k = midpoint(kijun)
    for i in range(n):
        a = t[i]
        b = k[i]
        if a is not None and b is not None:
            span_a[i] = (a + b) / 2.0
        if i >= senkou_b - 1:
            w = candles[i - senkou_b + 1 : i + 1]
            span_b[i] = (max(c.high for c in w) + min(c.low for c in w)) / 2.0
        if i + kijun < n:
            chikou[i + kijun] = candles[i].close
    return {
        "tenkan": t,
        "kijun": k,
        "senkou_a": span_a,
        "senkou_b": span_b,
        "chikou": chikou,
    }


def vwap(candles: Sequence[Candle]) -> list[Optional[float]]:
    """Anchored VWAP from the first bar of the series (session-anchored in live use)."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    pv = 0.0
    vol = 0.0
    for i, c in enumerate(candles):
        v = c.volume or 0.0
        pv += c.typical * v
        vol += v
        out[i] = (pv / vol) if vol > 0 else None
    return out


def hma(values: Sequence[Optional[float]], period: int = 16) -> list[Optional[float]]:
    """Hull Moving Average: wma(wma(n/2)*2 - wma(n), sqrt(n))."""
    half = max(1, period // 2)
    root = max(1, int(round(math.sqrt(period))))
    w_half = wma(values, half)
    w_full = wma(values, period)
    raw = [
        (2.0 * a - b) if (a is not None and b is not None) else None for a, b in zip(w_half, w_full)
    ]
    return wma(raw, root)


def pivots(candles: Sequence[Candle]) -> dict:
    """Classic floor-trader pivot levels from the previous completed bar."""
    n = len(candles)
    r1: list[Optional[float]] = [None] * n
    s1: list[Optional[float]] = [None] * n
    r2: list[Optional[float]] = [None] * n
    s2: list[Optional[float]] = [None] * n
    for i in range(1, n):
        p = candles[i - 1]
        pivot = (p.high + p.low + p.close) / 3.0
        r1[i] = 2 * pivot - p.low
        s1[i] = 2 * pivot - p.high
        r2[i] = pivot + (p.high - p.low)
        s2[i] = pivot - (p.high - p.low)
    return {"pivot": [None] + [((candles[i - 1].high + candles[i - 1].low + candles[i - 1].close) / 3.0) for i in range(1, n)], "r1": r1, "s1": s1, "r2": r2, "s2": s2}


def linear_regression(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Rolling least-squares endpoint (the value of the fitted line at bar i)."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if any(v is None for v in window):
            continue
        ys = [float(v) for v in window]
        xs = list(range(period))
        mx = sum(xs) / period
        my = sum(ys) / period
        num = sum((xs[j] - mx) * (ys[j] - my) for j in range(period))
        den = sum((xs[j] - mx) ** 2 for j in range(period))
        if den == 0:
            continue
        slope = num / den
        out[i] = my + slope * (period - 1 - mx)
    return out


def donchian(candles: Sequence[Candle], period: int = 20) -> dict:
    """Donchian channels: upper, middle, lower."""
    n = len(candles)
    upper: list[Optional[float]] = [None] * n
    lower: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        w = candles[i - period + 1 : i + 1]
        hi = max(c.high for c in w)
        lo = min(c.low for c in w)
        upper[i] = hi
        lower[i] = lo
    mid = [
        ((a + b) / 2.0) if (a is not None and b is not None) else None
        for a, b in zip(upper, lower)
    ]
    return {"upper": upper, "mid": mid, "lower": lower}
