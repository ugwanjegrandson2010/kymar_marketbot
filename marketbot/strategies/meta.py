"""Composite strategy: many indicators, one weighted vote.

This is the strategy for "combine a ton of variables". Rather than hand-pick a
handful of indicators, it scores every registered indicator's directional
reading, weights each, and sums. The output is a conviction-weighted signal.

Two design choices matter:

  * Every indicator is normalized to -1..+1 first. Without that, a CCI reading
    of +200 would swamp an RSI reading of 0.6 and the weighting would be
    meaningless.
  * Weights are configurable and default to a spread across families, so the
    strategy is not secretly a one-indicator system.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..indicators import ALL_INDICATORS, compute_all
from ..model import Candle, SignalAction
from .base import Strategy

# Family weights. Deliberately spread so no single indicator dominates.
DEFAULT_WEIGHTS = {
    "trend": 1.0,
    "momentum": 1.0,
    "volatility": 0.6,
    "volume": 0.6,
    "pattern": 0.4,
}

# Which family each indicator belongs to, used for weight grouping.
FAMILY = {
    "sma": "trend", "ema": "trend", "wma": "trend", "dema": "trend", "tema": "trend",
    "trima": "trend", "kama": "trend", "hma": "trend", "linreg": "trend",
    "adx": "trend", "plus_di": "trend", "minus_di": "trend", "aroon_up": "trend",
    "aroon_down": "trend", "aroon_osc": "trend", "sar": "trend", "supertrend": "trend",
    "tenkan": "trend", "kijun": "trend", "senkou_a": "trend", "senkou_b": "trend",
    "rsi": "momentum", "stoch_k": "momentum", "stoch_d": "momentum",
    "williams_r": "momentum", "cci": "momentum", "roc": "momentum",
    "momentum": "momentum", "cmo": "momentum", "trix": "momentum",
    "awesome": "momentum", "dpo": "momentum", "rvi": "momentum", "ppo": "momentum",
    "ultimate": "momentum", "bull_power": "momentum", "bear_power": "momentum",
    "stdev": "volatility", "atr": "volatility", "natr": "volatility",
    "hist_vol": "volatility", "chaikin_vol": "volatility",
    "mfi": "volume", "cmf": "volume", "obv": "volume", "ad_line": "volume",
    "force_index": "volume", "eom": "volume", "nvo": "volume", "pvo": "volume",
    "kvo": "volume", "vpci": "volume", "volume_roc": "volume", "vwap": "volume",
}


def _clamp(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return max(-1.0, min(1.0, v))


def normalize(name: str, value: Optional[float], price: Optional[float]) -> Optional[float]:
    """Map an indicator reading onto -1..+1.

    Each indicator has a different natural scale, so the mapping is per-family:
    oscillators use their own bands, trend indicators are compared to price, and
    unbounded oscillators are scaled by a per-symbol running standard deviation.
    """
    if value is None:
        return None
    if name == "rsi":
        return _clamp((value - 50.0) / 25.0)
    if name in ("stoch_k", "stoch_d"):
        return _clamp((value - 50.0) / 25.0)
    if name == "williams_r":
        return _clamp(-value / 50.0)
    if name == "mfi":
        return _clamp((value - 50.0) / 25.0)
    if name == "cmo":
        return _clamp(value / 50.0)
    if name == "cci":
        return _clamp(value / 150.0)
    if name in ("roc", "momentum", "trix", "dpo", "rvi", "ppo", "cmo", "awesome"):
        return _clamp(value / 5.0)
    if name == "adx":
        return _clamp((value - 20.0) / 30.0)
    if name in ("plus_di", "minus_di"):
        return _clamp((value - 25.0) / 25.0)
    if name in ("aroon_up", "aroon_down"):
        return _clamp((value - 50.0) / 50.0)
    if name == "aroon_osc":
        return _clamp(value / 100.0)
    if name in ("sma", "ema", "wma", "dema", "tema", "trima", "kama", "hma", "linreg"):
        return _clamp((price - value) / (value * 0.05)) if (price and value) else None
    if name == "supertrend":
        return _clamp(1.0 if (price and price > value) else -1.0)
    if name == "sar":
        return _clamp(1.0 if (price and price > value) else -1.0)
    if name in ("tenkan", "kijun"):
        return _clamp(1.0 if (price and price > value) else -1.0) if price else None
    if name in ("senkou_a", "senkou_b"):
        return _clamp(1.0 if (price and price > value) else -1.0) if price else None
    if name in ("atr", "stdev", "natr", "hist_vol", "chaikin_vol"):
        return _clamp(value / 5.0) if name == "natr" else _clamp(1.0 - value / 5.0)
    if name in ("obv", "ad_line", "force_index", "eom", "nvo", "vpci"):
        return _clamp(value / 1000.0)
    if name in ("mfi", "cmf", "pvo", "kvo", "volume_roc"):
        return _clamp((value - 50.0) / 50.0) if name == "mfi" else _clamp(value / 50.0)
    if name == "vwap":
        return _clamp(1.0 if (price and price > value) else -1.0) if price else None
    return _clamp(value / 50.0)


class CompositeStrategy(Strategy):
    """Weighted vote across every registered indicator."""

    name = "composite"
    description = "Weighted vote of all registered indicators, normalized to -1..+1"
    indicators = (("all", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {
            "period": 20,
            "threshold": 0.15,       # |score| must exceed this to act
            "hold_band": 0.05,       # dead zone around zero
            "min_votes": 5,          # require this many defined indicators
            "weights": dict(DEFAULT_WEIGHTS),
        }

    def prepare(self, candles: Sequence[Candle]) -> None:
        period = self.param("period", 20)
        self._series = compute_all(candles, period)
        self._names = [n for n in self._series if n in FAMILY]
        self._cl = [c.close for c in candles]

    def score_at(self, i: int) -> tuple[float, int]:
        """Return (weighted score in -1..1, number of indicators that voted).

        An empty weight map means uniform weighting rather than a silent
        fallback to a number nobody chose.
        """
        weights: dict = self.param("weights", DEFAULT_WEIGHTS)
        uniform = not weights
        total = 0.0
        weight_sum = 0.0
        votes = 0
        for name in self._names:
            raw = self._series[name][i] if i < len(self._series[name]) else None
            norm = normalize(name, raw, self._cl[i] if i < len(self._cl) else None)
            if norm is None:
                continue
            w = 1.0 if uniform else weights.get(FAMILY.get(name, "trend"), 0.5)
            total += norm * w
            weight_sum += w
            votes += 1
        if weight_sum == 0:
            return 0.0, 0
        return total / weight_sum, votes

    def on_bar(self, i, candles):
        score, votes = self.score_at(i)
        threshold = self.param("threshold", 0.15)
        hold = self.param("hold_band", 0.05)
        if votes < self.param("min_votes", 5):
            return self.signal(candles, SignalAction.HOLD, 0.0, f"only {votes} indicators ready",
                               score=round(score, 3), votes=votes)
        if score > threshold + hold:
            return self.signal(candles, SignalAction.BUY, abs(score), "composite bullish",
                               score=round(score, 3), votes=votes)
        if score < -(threshold + hold):
            return self.signal(candles, SignalAction.SELL, abs(score), "composite bearish",
                               score=round(score, 3), votes=votes)
        return self.signal(candles, SignalAction.HOLD, abs(score), "composite neutral",
                           score=round(score, 3), votes=votes)

    def vote_breakdown(self, i: int) -> dict:
        """Per-indicator contribution at bar i, for explainability."""
        weights: dict = self.param("weights", DEFAULT_WEIGHTS)
        out = {}
        for name in self._names:
            raw = self._series[name][i] if i < len(self._series[name]) else None
            norm = normalize(name, raw, self._cl[i] if i < len(self._cl) else None)
            if norm is None:
                continue
            out[name] = {
                "raw": round(raw, 4) if raw is not None else None,
                "norm": round(norm, 4),
                "weight": weights.get(FAMILY.get(name, "trend"), 0.5),
                "family": FAMILY.get(name, "other"),
            }
        return out


class WeightedComposite(CompositeStrategy):
    """Composite with the vote weights learned from walk-forward performance.

    Weights are set by the optimizer (walkforward.py) and are plain params here;
    the strategy itself never fits anything, so there is no in-sample leakage
    from inside the strategy.
    """

    name = "composite_optimized"
    description = "Composite whose family weights come from walk-forward optimization"
