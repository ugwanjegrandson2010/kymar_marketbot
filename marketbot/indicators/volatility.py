"""Volatility indicators and band constructs."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..model import Candle
from .core import atr, closes, sma, stdev


def bollinger(
    values: Sequence[Optional[float]], period: int = 20, num_std: float = 2.0
) -> dict:
    """Bollinger Bands plus %B and bandwidth, the three readings traders use."""
    n = len(values)
    mid = sma(values, period)
    sd = stdev(values, period)
    upper: list[Optional[float]] = [None] * n
    lower: list[Optional[float]] = [None] * n
    pct_b: list[Optional[float]] = [None] * n
    bandwidth: list[Optional[float]] = [None] * n
    for i in range(n):
        m, s = mid[i], sd[i]
        if m is None or s is None:
            continue
        u, l = m + num_std * s, m - num_std * s
        upper[i], lower[i] = u, l
        v = values[i]
        if v is not None and u != l:
            pct_b[i] = (float(v) - l) / (u - l)
            bandwidth[i] = (u - l) / m if m else None
    return {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth}


def keltner(
    candles: Sequence[Candle], period: int = 20, atr_period: int = 10, multiplier: float = 2.0
) -> dict:
    """Keltner Channels: EMA basis with ATR-based band width."""
    n = len(candles)
    cl = closes(candles)
    from .core import ema

    mid = ema(cl, period)
    a = atr(candles, atr_period)
    upper: list[Optional[float]] = [None] * n
    lower: list[Optional[float]] = [None] * n
    for i in range(n):
        if mid[i] is None or a[i] is None:
            continue
        upper[i] = mid[i] + multiplier * a[i]
        lower[i] = mid[i] - multiplier * a[i]
    return {"upper": upper, "mid": mid, "lower": lower}


def natr(values: Sequence[Optional[float]], candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """ATR as a percent of price, which makes volatility comparable across assets."""
    a = atr(candles, period)
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        av, v = a[i], values[i]
        if av is None or v is None or v == 0:
            continue
        out[i] = 100.0 * av / float(v)
    return out


def historical_volatility(
    candles: Sequence[Candle], period: int = 20, annualize: bool = True
) -> list[Optional[float]]:
    """Annualized (by default) standard deviation of log returns."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    rets: list[Optional[float]] = [None] * n
    for i in range(1, n):
        p, c = candles[i - 1].close, candles[i].close
        if p <= 0 or c <= 0:
            continue
        rets[i] = math.log(c / p)
    sd = stdev(rets, period)
    bars_per_year = 365.0 if annualize else 1.0
    for i in range(n):
        if sd[i] is None:
            continue
        out[i] = sd[i] * math.sqrt(bars_per_year)
    return out


def chaikin_volatility(candles: Sequence[Candle], period: int = 10) -> list[Optional[float]]:
    """Chaikin's volatility: rate of change of the range (H-L) moving average."""
    n = len(candles)
    rng = [c.high - c.low for c in candles]
    ma = sma(rng, period)
    out: list[Optional[float]] = [None] * n
    for i in range(period, n):
        a, b = ma[i], ma[i - period]
        if a is None or b is None or b == 0:
            continue
        out[i] = 100.0 * (a - b) / b
    return out


def squeeze(candles: Sequence[Candle], bb_period: int = 20, kc_period: int = 20) -> dict:
    """Bollinger/Keltner 'squeeze': volatility compression before expansion.

    A squeeze (Bollinger inside Keltner) is a common volatility-breakout trigger.
    """
    bb = bollinger(closes(candles), bb_period)
    kc = keltner(candles, kc_period)
    n = len(candles)
    in_squeeze: list[Optional[bool]] = [None] * n
    for i in range(n):
        bu, bl = bb["upper"][i], bb["lower"][i]
        ku, kl = kc["upper"][i], kc["lower"][i]
        if None in (bu, bl, ku, kl):
            continue
        in_squeeze[i] = bu < ku and bl > kl
    momentum_hist = _macd_hist(closes(candles))
    return {"squeeze": in_squeeze, "momentum": momentum_hist, "bb": bb, "kc": kc}


def _macd_hist(values):
    from .core import ema

    ef, es = ema(values, 12), ema(values, 26)
    line = [(a - b) if (a is not None and b is not None) else None for a, b in zip(ef, es)]
    sig = ema(line, 9)
    return [(a - b) if (a is not None and b is not None) else None for a, b in zip(line, sig)]


def stdev_rank(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Percentile rank of the current price within its own trailing range."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        w = values[i - period + 1 : i + 1]
        if any(v is None for v in w):
            continue
        cur = float(values[i])
        lo, hi = min(float(v) for v in w), max(float(v) for v in w)
        out[i] = 100.0 * (cur - lo) / (hi - lo) if hi != lo else 50.0
    return out
