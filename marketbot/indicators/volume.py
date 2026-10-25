"""Volume and money-flow indicators.

Volume data is only meaningful where it exists. CoinGecko's market_chart returns
it; its OHLC endpoint does not. Providers that supply a flat/zero volume are
flagged by `has_volume()` so a strategy can avoid pretending.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..model import Candle
from .core import closes, ema, sma


def has_volume(candles: Sequence[Candle]) -> bool:
    """True when at least a third of bars carry non-zero volume."""
    if not candles:
        return False
    nonzero = sum(1 for c in candles if (c.volume or 0.0) > 0)
    return nonzero >= len(candles) / 3


def obv(candles: Sequence[Candle]) -> list[Optional[float]]:
    """On-Balance Volume: cumulative signed volume."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    if n == 0:
        return out
    total = 0.0
    for i in range(n):
        if i > 0:
            prev, cur = candles[i - 1].close, candles[i].close
            if cur > prev:
                total += candles[i].volume or 0.0
            elif cur < prev:
                total -= candles[i].volume or 0.0
        out[i] = total
    return out


def ad_line(candles: Sequence[Candle]) -> list[Optional[float]]:
    """Accumulation/Distribution line (Close Location Value based)."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    total = 0.0
    for i, c in enumerate(candles):
        denom = (c.high - c.low)
        clv = ((c.close - c.low) - (c.high - c.close)) / denom if denom else 0.0
        total += clv * (c.volume or 0.0)
        out[i] = total
    return out


def mfi(candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Money Flow Index: RSI applied to typical price times volume."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    if n <= period:
        return out
    tp = [c.typical for c in candles]
    pos = [0.0] * n
    neg = [0.0] * n
    for i in range(1, n):
        d = tp[i] - tp[i - 1]
        flow = d * (candles[i].volume or 0.0)
        if d > 0:
            pos[i] = flow
        elif d < 0:
            neg[i] = -flow
    for i in range(period, n):
        pos_sum = sum(pos[i - period + 1 : i + 1])
        neg_sum = sum(neg[i - period + 1 : i + 1])
        if neg_sum == 0:
            # No down-flow in the window: the index saturates at 100 rather than
            # dividing by zero. Mirrors RSI's convention.
            out[i] = 100.0 if pos_sum > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1 + pos_sum / neg_sum)
    return out


def chaikin_money_flow(candles: Sequence[Candle], period: int = 20) -> list[Optional[float]]:
    """Chaikin Money Flow: AD-line flow normalized by total volume."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        w = candles[i - period + 1 : i + 1]
        num = 0.0
        den = 0.0
        for c in w:
            denom = c.high - c.low
            clv = ((c.close - c.low) - (c.high - c.close)) / denom if denom else 0.0
            v = c.volume or 0.0
            num += clv * v
            den += v
        out[i] = num / den if den else 0.0
    return out


def vwap_session(candles: Sequence[Candle]) -> list[Optional[float]]:
    """Session-anchored VWAP: resets at each UTC date boundary."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    import datetime as _dt

    pv = 0.0
    vol = 0.0
    cur_day = None
    for i, c in enumerate(candles):
        day = _dt.datetime.fromtimestamp(c.ts, _dt.timezone.utc).date()
        if day != cur_day:
            cur_day = day
            pv = 0.0
            vol = 0.0
        v = c.volume or 0.0
        pv += c.typical * v
        vol += v
        out[i] = (pv / vol) if vol > 0 else None
    return out


def vwap_bands(
    candles: Sequence[Candle], period: int = 20, num_std: float = 2.0
) -> dict:
    """VWAP with standard-deviation bands (common intraday construct)."""
    vw = vwap_session(candles)
    cl = closes(candles)
    n = len(candles)
    upper: list[Optional[float]] = [None] * n
    lower: list[Optional[float]] = [None] * n
    for i in range(n):
        if vw[i] is None:
            continue
        w = cl[max(0, i - period + 1) : i + 1]
        w = [x for x in w if x is not None]
        if len(w) < 2:
            continue
        mean = sum(w) / len(w)
        var = sum((x - mean) ** 2 for x in w) / len(w)
        sd = var ** 0.5
        upper[i] = vw[i] + num_std * sd
        lower[i] = vw[i] - num_std * sd
    return {"vwap": vw, "upper": upper, "lower": lower}


def volume_roc(candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Percent change in volume."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    for i in range(period, n):
        a = candles[i].volume or 0.0
        b = candles[i - period].volume or 0.0
        out[i] = 100.0 * (a - b) / b if b else None
    return out


def eom(candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Ease of Movement: price move scaled by volume, smoothed by SMA.

    Volume is in the denominator, so a zero-volume bar must be handled before
    dividing — otherwise a flat or halted market raises ZeroDivisionError.
    """
    n = len(candles)
    raw: list[Optional[float]] = [None] * n
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        dist = (c.high + c.low) / 2.0 - (p.high + p.low) / 2.0
        vol = c.volume or 0.0
        if vol <= 0:
            raw[i] = 0.0
            continue
        box = (c.high - c.low) / vol
        raw[i] = dist / box if box else 0.0
    return sma(raw, period)


def force_index(candles: Sequence[Candle], period: int = 13) -> list[Optional[float]]:
    """Elder's Force Index: change in price times volume, smoothed by EMA."""
    n = len(candles)
    raw: list[Optional[float]] = [None] * n
    for i in range(1, n):
        raw[i] = (candles[i].close - candles[i - 1].close) * (candles[i].volume or 0.0)
    return ema(raw, period)


def nvo(candles: Sequence[Candle], period: int = 100) -> list[Optional[float]]:
    """Negative Volume Index: accumulates returns on down-volume days only."""
    n = len(candles)
    out: list[Optional[float]] = [None] * n
    if n == 0:
        return out
    total = 1000.0
    out[0] = total
    for i in range(1, n):
        v, pv = candles[i].volume or 0.0, candles[i - 1].volume or 0.0
        if v < pv:
            r = (candles[i].close - candles[i - 1].close) / candles[i - 1].close
            total *= 1 + r
        out[i] = total
    return out


def pvo(candles: Sequence[Candle], fast: int = 12, slow: int = 26) -> list[Optional[float]]:
    """Percentage Volume Oscillator."""
    v = [c.volume or 0.0 for c in candles]
    ef, es = ema(v, fast), ema(v, slow)
    out: list[Optional[float]] = [None] * len(v)
    for i in range(len(v)):
        a, b = ef[i], es[i]
        if a is None or b is None or b == 0:
            continue
        out[i] = 100.0 * (a - b) / b
    return out


def vpci(candles: Sequence[Candle], period: int = 21) -> list[Optional[float]]:
    """Volume Price Confirmation Indicator: trend filter using volume-weighted price."""
    n = len(candles)
    cl = closes(candles)
    vw = [(cl[i] * (candles[i].volume or 0.0)) for i in range(n)]
    sma_cl = sma(cl, period)
    sma_vw = sma(vw, period)
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        a, b = sma_cl[i], sma_vw[i]
        if a is None or b is None:
            continue
        out[i] = b - a
    return out


def kvo(candles: Sequence[Candle], fast: int = 34, slow: int = 55) -> list[Optional[float]]:
    """Klinger Volume Oscillator (signal line included)."""
    n = len(candles)
    vf: list[Optional[float]] = [None] * n
    trend: Optional[float] = None
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        hlm = (c.high + c.low + c.close) / 3.0
        plm = (p.high + p.low + p.close) / 3.0
        t = 1.0 if hlm > plm else -1.0
        if trend is not None and t != trend:
            dm = c.high - c.low
        else:
            dm = c.high - c.low
        vol_ = c.volume or 0.0
        vf[i] = vol_ * t * abs(2.0 * dm / (c.high - c.low) - 1.0) if (c.high - c.low) else 0.0
        trend = t
    ef, es = ema(vf, fast), ema(vf, slow)
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        a, b = ef[i], es[i]
        if a is not None and b is not None:
            out[i] = a - b
    return out
