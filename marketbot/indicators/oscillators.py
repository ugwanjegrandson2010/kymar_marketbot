"""Momentum oscillators.

Every oscillator is normalized into a 0-100 (or -100..100) reading so strategies
can compare conviction across indicators with different natural scales.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..model import Candle
from .core import _align, atr, closes, ema, sma, stdev, true_range, wilder


def rsi(values: Sequence[Optional[float]], period: int = 14) -> list[Optional[float]]:
    """Relative Strength Index with Wilder smoothing (the standard, not SMA)."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if n <= period:
        return out
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        v, p = values[i], values[i - 1]
        if v is None or p is None:
            continue
        d = float(v) - float(p)
        gains[i] = d if d > 0 else 0.0
        losses[i] = -d if d < 0 else 0.0

    # Seed with a simple average, then Wilder-smooth (Wilder's own method).
    first = 0
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    first = period
    out[first] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(first + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def stoch(
    candles: Sequence[Candle], k_period: int = 14, d_period: int = 3, smooth: int = 3
) -> tuple[list, list]:
    """Stochastic %K (smoothed) and %D."""
    n = len(candles)
    raw_k: list[Optional[float]] = [None] * n
    for i in range(k_period - 1, n):
        w = candles[i - k_period + 1 : i + 1]
        hh = max(c.high for c in w)
        ll = min(c.low for c in w)
        if hh == ll:
            raw_k[i] = 50.0
        else:
            raw_k[i] = 100.0 * (candles[i].close - ll) / (hh - ll)
    k = sma(raw_k, smooth)
    d = sma(k, d_period)
    return k, d


def williams_r(candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Williams %R, -100..0 scale."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        w = candles[i - period + 1 : i + 1]
        hh = max(c.high for c in w)
        ll = min(c.low for c in w)
        if hh == ll:
            out[i] = -50.0
        else:
            out[i] = -100.0 * (hh - candles[i].close) / (hh - ll)
    return out


def cci(candles: Sequence[Candle], period: int = 20) -> list[Optional[float]]:
    """Commodity Channel Index."""
    n = len(candles)
    tp = [c.typical for c in candles]
    ma = sma(tp, period)
    out: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        if ma[i] is None:
            continue
        window = tp[i - period + 1 : i + 1]
        mean_dev = sum(abs(v - ma[i]) for v in window) / period
        if mean_dev == 0:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - ma[i]) / (0.015 * mean_dev)
    return out


def roc(values: Sequence[Optional[float]], period: int = 10) -> list[Optional[float]]:
    """Rate of change, percent."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(period, n):
        v, p = values[i], values[i - period]
        if v is None or p is None or p == 0:
            continue
        out[i] = 100.0 * (float(v) - float(p)) / float(p)
    return out


def momentum(values: Sequence[Optional[float]], period: int = 10) -> list[Optional[float]]:
    """Raw momentum (price minus price n bars ago)."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(period, n):
        v, p = values[i], values[i - period]
        if v is None or p is None:
            continue
        out[i] = float(v) - float(p)
    return out


def cmo(values: Sequence[Optional[float]], period: int = 14) -> list[Optional[float]]:
    """Chande Momentum Oscillator, -100..100."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    gains = 0.0
    losses = 0.0
    for i in range(1, min(n, period + 1)):
        d = _d(values, i)
        if d is None:
            return out
        if d > 0:
            gains += d
        else:
            losses -= d
    if gains + losses == 0:
        return out
    out[period] = 100.0 * (gains - losses) / (gains + losses)
    for i in range(period + 1, n):
        d = _d(values, i)
        dp = _d(values, i - period)
        if d is None or dp is None:
            continue
        gains += max(d, 0.0) - max(dp, 0.0)
        losses += max(-d, 0.0) - max(-dp, 0.0)
        total = gains + losses
        out[i] = 0.0 if total == 0 else 100.0 * (gains - losses) / total
    return out


def trix(values: Sequence[Optional[float]], period: int = 14) -> list[Optional[float]]:
    """TRIX: percent rate of change of a triple-smoothed EMA."""
    e1 = ema(values, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(1, n):
        a, b = e3[i], e3[i - 1]
        if a is None or b is None or b == 0:
            continue
        out[i] = 100.0 * (a - b) / b
    return out


def ultimate_oscillator(
    candles: Sequence[Candle], p1: int = 7, p2: int = 14, p3: int = 28
) -> list[Optional[float]]:
    """Larry Williams' Ultimate Oscillator, 0-100."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    tr = true_range(candles)
    bp: list[Optional[float]] = [None] * n
    for i in range(1, n):
        bp[i] = candles[i].close - min(candles[i].low, candles[i - 1].close)

    def avg(num_seq: Sequence[Optional[float]], den_seq: Sequence[Optional[float]], period: int):
        res: list[Optional[float]] = [None] * n
        for i in range(period, n):
            num_window = num_seq[i - period + 1 : i + 1]
            den_window = den_seq[i - period + 1 : i + 1]
            if any(v is None for v in num_window) or any(v is None for v in den_window):
                continue
            den = sum(float(v) for v in den_window)
            res[i] = sum(float(v) for v in num_window) / den if den else None
        return res

    a7 = avg(bp, tr, p1)
    a14 = avg(bp, tr, p2)
    a28 = avg(bp, tr, p3)
    for i in range(n):
        if a7[i] is None or a14[i] is None or a28[i] is None:
            continue
        denom = 4 + 2 + 1
        out[i] = 100.0 * (4 * a7[i] + 2 * a14[i] + a28[i]) / denom
    return out


def awesome_oscillator(candles: Sequence[Candle], fast: int = 5, slow: int = 34) -> list[Optional[float]]:
    """Bill Williams' Awesome Oscillator: SMA(median price) fast minus slow."""
    median = [(c.high + c.low) / 2.0 for c in candles]
    return [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(sma(median, fast), sma(median, slow))
    ]


def kst(
    values: Sequence[Optional[float]],
    periods: tuple = (10, 10, 10, 15),
    smooth: tuple = (10, 10, 10, 15),
    signal: int = 9,
) -> tuple[list, list]:
    """Know Sure Thing oscillator (Martin Pring) and its signal line."""
    weighted: list[Optional[float]] = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            weighted[i] = None
    acc: list[Optional[float]] = [0.0 if v is not None else None for v in values]
    for p, s in zip(periods, smooth):
        r = roc(values, p)
        sm = sma(r, s)
        for i in range(len(values)):
            a, b = acc[i], sm[i]
            acc[i] = (a + b) if (a is not None and b is not None) else None
    return acc, sma(acc, signal)


def dpo(values: Sequence[Optional[float]], period: int = 20) -> list[Optional[float]]:
    """Detrended Price Oscillator: price minus SMA shifted back by half the period."""
    n = len(values)
    ma = sma(values, period)
    shift = (period // 2) + 1
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        j = i - shift
        v, m = values[i], ma[j] if j >= 0 else None
        if v is None or m is None:
            continue
        out[i] = float(v) - float(m)
    return out


def fisher_transform(
    values: Sequence[Optional[float]], period: int = 10, smooth: int = 3
) -> tuple[list, list]:
    """Ehlers' Fisher Transform plus trigger line. Normalized price -> Gaussian."""
    n = len(values)
    highest: list[Optional[float]] = [None] * n
    lowest: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        w = values[i - period + 1 : i + 1]
        if any(v is None for v in w):
            continue
        highest[i] = max(float(v) for v in w)
        lowest[i] = min(float(v) for v in w)

    raw: list[Optional[float]] = [0.0] * n
    for i in range(period - 1, n):
        hi, lo = highest[i], lowest[i]
        if hi is None or lo is None or hi == lo:
            continue
        raw[i] = 0.5 * (2 * ((float(values[i]) - lo) / (hi - lo) - 0.5)) + 0.5 * (
            raw[i - 1] if i else 0.0
        )
        raw[i] = max(-0.999, min(0.999, raw[i]))

    fish: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        r = raw[i]
        if r is None:
            continue
        prev = fish[i - 1] if i and fish[i - 1] is not None else 0.0
        fish[i] = 0.5 * math.log((1 + r) / (1 - r)) + 0.5 * prev
    trigger = [None] * n
    for i in range(1, n):
        if fish[i] is not None and fish[i - 1] is not None:
            trigger[i] = fish[i - 1]
    return fish, trigger


def elder_ray(candles: Sequence[Candle], period: int = 13) -> dict:
    """Elder's Bull Power and Bear Power against an EMA."""
    e = ema([c.close for c in candles], period)
    bull: list[Optional[float]] = [None] * len(candles)
    bear: list[Optional[float]] = [None] * len(candles)
    for i, c in enumerate(candles):
        if e[i] is None:
            continue
        bull[i] = c.high - e[i]
        bear[i] = c.low - e[i]
    return {"bull": bull, "bear": bear, "ema": e}


def ppo(
    values: Sequence[Optional[float]], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list, list, list]:
    """Percentage Price Oscillator: scale-free MACD, in percent."""
    ef = ema(values, fast)
    es = ema(values, slow)
    line: list[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        a, b = ef[i], es[i]
        if a is None or b is None or b == 0:
            continue
        line[i] = 100.0 * (a - b) / abs(b)
    sig = ema(line, signal)
    hist = [
        (a - b) if (a is not None and b is not None) else None for a, b in zip(line, sig)
    ]
    return line, sig, hist


def rvi(candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Relative Vigor Index (Elder): close-open over high-low, smoothed."""
    n = len(candles)
    num = [0.0] * n
    den = [0.0] * n
    for i in range(n):
        c = candles[i]
        pc = candles[i - 1].close if i else c.close
        ph = candles[i - 1].high if i else c.high
        pl = candles[i - 1].low if i else c.low
        num[i] = (c.close - c.open) / 3.0 + (pc - candles[i - 1].open) / 3.0 if i else 0.0
        den[i] = (c.high - c.low) / 3.0 + (ph - pl) / 3.0 if i else 0.0
    s_num = _rolling_sum(num, period)
    s_den = _rolling_sum(den, period)
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        if s_den[i] is None or s_den[i] == 0 or s_num[i] is None:
            continue
        out[i] = 100.0 * s_num[i] / s_den[i]
    return out


def _rolling_sum(values: Sequence[float], period: int) -> list[Optional[float]]:
    n = len(values)
    out: list[Optional[float]] = [None] * n
    run = 0.0
    for i in range(n):
        run += values[i]
        if i >= period:
            run -= values[i - period]
        if i >= period - 1:
            out[i] = run
    return out


def _d(values: Sequence[Optional[float]], i: int) -> Optional[float]:
    if i <= 0:
        return None
    v, p = values[i], values[i - 1]
    if v is None or p is None:
        return None
    return float(v) - float(p)
