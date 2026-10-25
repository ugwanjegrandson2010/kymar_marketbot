"""Indicator library: public API and a name registry.

`INDICATORS` maps a stable name to a callable so strategies, the CLI, and the
API can select indicators by string without importing every module.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..model import Candle
from .core import (  # noqa: E402
    adx,
    aroon,
    atr,
    closes,
    donchian,
    ema,
    hma,
    ichimoku,
    linear_regression,
    macd,
    pivots,
    parabolic_sar,
    sma,
    stdev,
    supertrend,
    true_range,
    vwap,
    wilder,
    wma,
)
from .oscillators import (
    awesome_oscillator,
    cci,
    cmo,
    dpo,
    elder_ray,
    fisher_transform,
    kst,
    momentum,
    ppo,
    roc,
    rsi,
    rvi,
    stoch,
    trix,
    ultimate_oscillator,
    williams_r,
)
from .patterns import all_patterns
from .volatility import (
    bollinger,
    chaikin_volatility,
    historical_volatility,
    keltner,
    natr,
    squeeze,
    stdev_rank,
)
from .volume import (
    ad_line,
    chaikin_money_flow,
    eom,
    force_index,
    has_volume,
    kvo,
    mfi,
    nvo,
    obv,
    pvo,
    vpci,
    volume_roc,
    vwap_bands,
    vwap_session,
)


def dema(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Double Exponential Moving Average, which removes most of EMA's lag."""
    e1 = ema(values, period)
    e2 = ema(e1, period)
    out: list[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        a, b = e1[i], e2[i]
        if a is None or b is None:
            continue
        out[i] = 2 * a - b
    return out


def tema(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Triple Exponential Moving Average: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = ema(values, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    out: list[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        a, b, c = e1[i], e2[i], e3[i]
        if a is None or b is None or c is None:
            continue
        out[i] = 3 * a - 3 * b + c
    return out


def trima(values: Sequence[Optional[float]], period: int) -> list[Optional[float]]:
    """Triangular Moving Average: SMA of SMA, the classic Gann-style average."""
    return sma(sma(values, period), period)


def kama(
    values: Sequence[Optional[float]], period: int = 10, fast: int = 2, slow: int = 30
) -> list[Optional[float]]:
    """Kaufman Adaptive Moving Average, which self-tunes to noise."""
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    change = [None] * n
    volatility = [None] * n
    for i in range(period, n):
        v, p = values[i], values[i - period]
        if v is None or p is None:
            continue
        change[i] = abs(float(v) - float(p))
        volatility[i] = sum(
            abs(float(values[j]) - float(values[j - 1]))
            for j in range(i - period + 1, i + 1)
            if values[j] is not None and values[j - 1] is not None
        )
    prev: Optional[float] = None
    for i in range(n):
        ch, vol = change[i], volatility[i]
        if ch is None or vol is None or vol == 0:
            continue
        er = ch / vol
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        if prev is None:
            prev = float(values[i])
        else:
            prev = prev + sc * (float(values[i]) - prev)
        out[i] = prev
    return out


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
# Every entry takes (candles, period) and returns a list aligned to the input,
# so `compute`/`compute_all` need no per-indicator special casing. CLOSE_INDICATORS
# are wrapped to extract closes so their underlying functions stay pure math on
# a float sequence.
def _closes_fn(fn):
    return lambda candles, period: fn(closes(candles), period)


def _ohlc_fn(fn):
    return lambda candles, period: fn(candles, period)


CLOSE_INDICATORS: dict[str, Callable] = {
    "sma": _closes_fn(sma),
    "ema": _closes_fn(ema),
    "wma": _closes_fn(wma),
    "dema": _closes_fn(dema),
    "tema": _closes_fn(tema),
    "trima": _closes_fn(trima),
    "kama": _closes_fn(kama),
    "hma": _closes_fn(hma),
    "stdev": _closes_fn(stdev),
    "rsi": _closes_fn(rsi),
    "roc": _closes_fn(roc),
    "momentum": _closes_fn(momentum),
    "cmo": _closes_fn(cmo),
    "trix": _closes_fn(trix),
    "dpo": _closes_fn(dpo),
    "linreg": _closes_fn(linear_regression),
    "stoch_k": lambda c, p: stoch(c, p)[0],
    "stoch_d": lambda c, p: stoch(c, p)[1],
    "williams_r": lambda c, p: williams_r(c, p),
    "cci": lambda c, p: cci(c, p),
    "awesome": lambda c, p: awesome_oscillator(c, p),
    "rvi": lambda c, p: rvi(c, p),
    "ppo": lambda c, p: ppo(closes(c), p)[0],
    # These three were defined and imported but never registered, so they were
    # invisible to `compute` and absent from `available()` — and the CLI's
    # default list still asked for "macd", which raised KeyError.
    # macd's own `fast`/`slow`/`signal` are its real knobs, so `period` must not
    # be forwarded positionally: it would silently rebind `fast`.
    "macd": _closes_fn(lambda v, p: macd(v)[0]),
    "macd_signal": _closes_fn(lambda v, p: macd(v)[1]),
    "macd_hist": _closes_fn(lambda v, p: macd(v)[2]),
    "kst": _closes_fn(lambda v, p: kst(v)[0]),
    "kst_signal": _closes_fn(lambda v, p: kst(v)[1]),
    "fisher": _closes_fn(lambda v, p: fisher_transform(v, p)[0]),
    "fisher_trigger": _closes_fn(lambda v, p: fisher_transform(v, p)[1]),
}

# callables taking (candles, params) -> list aligned to input.
# Each entry names its own default period explicitly: a positional `period`
# argument would otherwise silently rebind the wrong parameter of functions
# with more than one tuning knob (e.g. kvo's fast/slow).
OHLC_INDICATORS: dict[str, Callable] = {
    "atr": lambda c, p: atr(c, p),
    "adx": lambda c, p: adx(c, p)[0],
    "plus_di": lambda c, p: adx(c, p)[1],
    "minus_di": lambda c, p: adx(c, p)[2],
    "aroon_up": lambda c, p: aroon(c, p)[0],
    "aroon_down": lambda c, p: aroon(c, p)[1],
    "aroon_osc": lambda c, p: aroon(c, p)[2],
    "sar": lambda c, p: parabolic_sar(c),
    "supertrend": lambda c, p: supertrend(c, 10, 3.0),
    "vwap": lambda c, p: vwap(c),
    "mfi": lambda c, p: mfi(c, p),
    "cmf": lambda c, p: chaikin_money_flow(c, p),
    "obv": lambda c, p: obv(c),
    "ad_line": lambda c, p: ad_line(c),
    "force_index": lambda c, p: force_index(c, p),
    "eom": lambda c, p: eom(c, p),
    "nvo": lambda c, p: nvo(c, p),
    "pvo": lambda c, p: pvo(c),
    "kvo": lambda c, p: kvo(c),
    "vpci": lambda c, p: vpci(c, p),
    "volume_roc": lambda c, p: volume_roc(c, p),
    "chaikin_vol": lambda c, p: chaikin_volatility(c, p),
    "hist_vol": lambda c, p: historical_volatility(c, p),
    "natr": lambda c, p: natr(closes(c), c, p),
    "ultimate": lambda c, p: ultimate_oscillator(c),
    "bull_power": lambda c, p: elder_ray(c, p)["bull"],
    "bear_power": lambda c, p: elder_ray(c, p)["bear"],
    "tenkan": lambda c, p: ichimoku(c)["tenkan"],
    "kijun": lambda c, p: ichimoku(c)["kijun"],
    "senkou_a": lambda c, p: ichimoku(c)["senkou_a"],
    "senkou_b": lambda c, p: ichimoku(c)["senkou_b"],
    "chikou": lambda c, p: ichimoku(c)["chikou"],
}

# callables taking (candles, params) -> dict of aligned series
BAND_INDICATORS: dict[str, Callable] = {
    "bollinger": lambda c, p: bollinger(closes(c), p),
    "keltner": lambda c, p: keltner(c, p),
    "donchian": lambda c, p: donchian(c, p),
    "vwap_bands": lambda c, p: vwap_bands(c, p),
    "squeeze": lambda c, p: squeeze(c, p),
    "pivots": lambda c, p: pivots(c, p),
}

ALL_INDICATORS = {**CLOSE_INDICATORS, **OHLC_INDICATORS}


def available() -> list[str]:
    return sorted(ALL_INDICATORS.keys())


def compute(name: str, candles: Sequence[Candle], period: int = 14) -> list[Optional[float]]:
    """Compute one indicator by name, raising KeyError for unknown names."""
    if name not in ALL_INDICATORS:
        raise KeyError(f"unknown indicator {name!r}; available: {available()}")
    return ALL_INDICATORS[name](candles, period)


def compute_all(candles: Sequence[Candle], period: int = 14) -> dict[str, list]:
    """Compute every indicator at once, for the composite strategy."""
    out: dict[str, list] = {}
    for name, fn in ALL_INDICATORS.items():
        try:
            out[name] = fn(candles, period)
        except Exception:
            out[name] = [None] * len(candles)
    return out


__all__ = [
    "ALL_INDICATORS",
    "BAND_INDICATORS",
    "CLOSE_INDICATORS",
    "OHLC_INDICATORS",
    "adx",
    "aroon",
    "atr",
    "bollinger",
    "closes",
    "compute",
    "compute_all",
    "dema",
    "donchian",
    "ema",
    "has_volume",
    "hma",
    "ichimoku",
    "kama",
    "keltner",
    "macd",
    "parabolic_sar",
    "pivots",
    "rsi",
    "sma",
    "squeeze",
    "stdev",
    "supertrend",
    "tema",
    "trima",
    "true_range",
    "vwap",
    "wma",
    "available",
]
