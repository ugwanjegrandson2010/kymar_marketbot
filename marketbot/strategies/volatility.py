"""Volatility strategies: band breakouts, band reversion, and squeeze expansion.

Band width is a volatility proxy, so these strategies also encode a volatility
regime view — a squeeze (Bollinger inside Keltner) precedes expansion, and
expansion is where breakouts pay.
"""

from __future__ import annotations

from ..indicators.volatility import bollinger, keltner, squeeze, stdev_rank
from ..indicators.core import atr as _atr
from ..model import SignalAction
from .base import Strategy


class BollingerBreakout(Strategy):
    """Buy a close above the upper band, sell below the lower band."""

    name = "bollinger_breakout"
    description = "Close beyond the Bollinger band continues in that direction"
    indicators = (("bollinger", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 20, "num_std": 2.0}

    def prepare(self, candles):
        self._bb = bollinger(
            [c.close for c in candles], self.param("period", 20), self.param("num_std", 2.0)
        )

    def on_bar(self, i, candles):
        u, l = self._bb["upper"][i], self._bb["lower"][i]
        if u is None or l is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        if price > u:
            return self.signal(candles, SignalAction.BUY, 1.0, "close above upper band", upper=u)
        if price < l:
            return self.signal(candles, SignalAction.SELL, 1.0, "close below lower band", lower=l)
        return self.signal(candles, SignalAction.HOLD, reason="inside bands")


class BollingerReversion(Strategy):
    """Fade band touches back toward the mean — the opposite trade, stated plainly."""

    name = "bollinger_reversion"
    description = "Close below lower band buys, above upper band sells, exit at mean"
    indicators = (("bollinger", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 20, "num_std": 2.0}

    def prepare(self, candles):
        self._bb = bollinger(
            [c.close for c in candles], self.param("period", 20), self.param("num_std", 2.0)
        )

    def on_bar(self, i, candles):
        u, m, l = self._bb["upper"][i], self._bb["mid"][i], self._bb["lower"][i]
        if None in (u, m, l):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        if price < l:
            return self.signal(candles, SignalAction.BUY, 1.0, "close below lower band", lower=l)
        if price > u:
            return self.signal(candles, SignalAction.SELL, 1.0, "close above upper band", upper=u)
        if abs(price - m) < (u - l) * 0.05:
            return self.signal(candles, SignalAction.CLOSE, 1.0, "returned to mean")
        return self.signal(candles, SignalAction.HOLD, reason="between band and mean")


class SqueezeBreakout(Strategy):
    """Bollinger/Keltner squeeze release: enter when volatility expands out."""

    name = "squeeze_breakout"
    description = "Trade the direction of the move when the Bollinger/Keltner squeeze releases"
    indicators = (("squeeze", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"bb_period": 20, "kc_period": 20}

    def prepare(self, candles):
        self._sq = squeeze(candles, self.param("bb_period", 20), self.param("kc_period", 20))

    def on_bar(self, i, candles):
        sq = self._sq["squeeze"][i]
        mom = self._sq["momentum"][i]
        if sq is None or mom is None or i < 1 or self._sq["squeeze"][i - 1] is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if self._sq["squeeze"][i - 1] and not sq:
            if mom > 0:
                return self.signal(candles, SignalAction.BUY, min(1.0, abs(mom) / 10 + 0.5),
                                   "squeeze released upward", momentum=mom)
            if mom < 0:
                return self.signal(candles, SignalAction.SELL, min(1.0, abs(mom) / 10 + 0.5),
                                   "squeeze released downward", momentum=mom)
        return self.signal(candles, SignalAction.HOLD, reason="squeeze holding")


class KeltnerBreakout(Strategy):
    """Keltner channel break with ATR conviction."""

    name = "keltner_breakout"
    description = "Close outside the Keltner channel continues in that direction"
    indicators = (("keltner", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 20, "atr_period": 10, "multiplier": 2.0}

    def prepare(self, candles):
        self._kc = keltner(
            candles, self.param("period", 20), self.param("atr_period", 10),
            self.param("multiplier", 2.0),
        )

    def on_bar(self, i, candles):
        u, l = self._kc["upper"][i], self._kc["lower"][i]
        if u is None or l is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        width = u - l
        if price > u:
            return self.signal(candles, SignalAction.BUY, min(1.0, (price - u) / width + 0.5),
                               "above Keltner upper", upper=u)
        if price < l:
            return self.signal(candles, SignalAction.SELL, min(1.0, (l - price) / width + 0.5),
                               "below Keltner lower", lower=l)
        return self.signal(candles, SignalAction.HOLD, reason="inside Keltner")


class StdevRankStrategy(Strategy):
    """Position within the trailing range: buy the top decile, sell the bottom."""

    name = "stdev_rank"
    description = "Buy when price is above 80% of its trailing range, sell below 20%"
    indicators = (("stdev_rank", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 20, "high": 80.0, "low": 20.0}

    def prepare(self, candles):
        self._sr = stdev_rank([c.close for c in candles], self.param("period", 20))

    def on_bar(self, i, candles):
        r = self._sr[i]
        if r is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if r >= self.param("high", 80.0):
            return self.signal(candles, SignalAction.BUY, r / 100.0, "top of range", rank=r)
        if r <= self.param("low", 20.0):
            return self.signal(candles, SignalAction.SELL, (100 - r) / 100.0, "bottom of range",
                               rank=r)
        return self.signal(candles, SignalAction.HOLD, reason="mid range")


class AtrVolatilityBreakout(Strategy):
    """ATR-multiple breakout: a move larger than N times ATR is a new trend."""

    name = "atr_breakout"
    description = "Bar range exceeding N times ATR starts a position in that direction"
    indicators = (("atr", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"atr_period": 14, "multiplier": 1.5}

    def prepare(self, candles):
        self._a = _atr(candles, self.param("atr_period", 14))

    def on_bar(self, i, candles):
        a = self._a[i]
        if a is None or a <= 0:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        bar_range = candles[i].high - candles[i].low
        mult = self.param("multiplier", 1.5)
        if bar_range > a * mult:
            if candles[i].close > candles[i].open:
                return self.signal(candles, SignalAction.BUY, 1.0, "expansive green bar",
                                   atr=a, bar_range=bar_range)
            return self.signal(candles, SignalAction.SELL, 1.0, "expansive red bar",
                               atr=a, bar_range=bar_range)
        return self.signal(candles, SignalAction.HOLD, reason="range within normal")
