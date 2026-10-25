"""Momentum / mean-reversion oscillator strategies.

Two distinct families live here and they must not be confused:
  * overbought/oversold reversion — fade extremes;
  * momentum continuation — follow the move.

Mixing them in one strategy without stating which is which is a common source of
a backtest that looks great and fails live, because the two regimes need
opposite behaviour.
"""

from __future__ import annotations

from ..indicators.core import atr
from ..indicators.oscillators import cci, stoch, williams_r
from ..indicators.oscillators import rsi as _rsi
from ..model import SignalAction
from .base import Strategy


class RsiMeanReversion(Strategy):
    """Fade RSI extremes: buy oversold, sell overbought, exit at neutrality."""

    name = "rsi_reversion"
    description = "Buy RSI below oversold, sell above overbought, exit at 50"
    indicators = (("rsi", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "oversold": 30, "overbought": 70, "exit": 50}

    def prepare(self, candles):
        self._r = _rsi([c.close for c in candles], self.param("period", 14))

    def on_bar(self, i, candles):
        r = self._r[i]
        if r is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        os_, ob, ex = self.param("oversold", 30), self.param("overbought", 70), self.param("exit", 50)
        prev = self._r[i - 1] if i > 0 else None
        if r < os_:
            return self.signal(candles, SignalAction.BUY, (os_ - r) / os_, "RSI oversold",
                               rsi=r)
        if r > ob:
            return self.signal(candles, SignalAction.SELL, (r - ob) / (100 - ob),
                               "RSI overbought", rsi=r)
        if prev is not None and prev < ex <= r:
            return self.signal(candles, SignalAction.CLOSE, 1.0, "RSI crossed up through exit")
        if prev is not None and prev > ex >= r:
            return self.signal(candles, SignalAction.CLOSE, 1.0, "RSI crossed down through exit")
        return self.signal(candles, SignalAction.HOLD, reason="RSI neutral")


class RsiTrendContinuation(Strategy):
    """RSI as a trend filter in the 40-60 band (Connors-style)."""

    name = "rsi_trend"
    description = "Buy when RSI rises through 55, sell when it falls through 45"
    indicators = (("rsi", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "buy": 55, "sell": 45}

    def prepare(self, candles):
        self._r = _rsi([c.close for c in candles], self.param("period", 14))

    def on_bar(self, i, candles):
        r = self._r[i]
        if r is None or i < 1 or self._r[i - 1] is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        prev = self._r[i - 1]
        if prev < self.param("buy", 55) <= r:
            return self.signal(candles, SignalAction.BUY, 0.8, "RSI momentum up", rsi=r)
        if prev > self.param("sell", 45) >= r:
            return self.signal(candles, SignalAction.SELL, 0.8, "RSI momentum down", rsi=r)
        return self.signal(candles, SignalAction.HOLD, reason="RSI flat")


class StochasticStrategy(Strategy):
    """Stochastic %K/%D cross inside an extreme zone."""

    name = "stochastic"
    description = "%K crossing %D in oversold goes long, in overbought goes short"
    indicators = (("stoch", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}

    def prepare(self, candles):
        self._k, self._d = stoch(candles, self.param("k_period", 14), self.param("d_period", 3))

    def on_bar(self, i, candles):
        k, d = self._k[i], self._d[i]
        if None in (k, d) or i < 1 or None in (self._k[i - 1], self._d[i - 1]):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        os_, ob = self.param("oversold", 20), self.param("overbought", 80)
        if self._k[i - 1] <= self._d[i - 1] and k > d and k < os_:
            return self.signal(candles, SignalAction.BUY, 1.0, "bullish stoch cross oversold",
                               k=k, d=d)
        if self._k[i - 1] >= self._d[i - 1] and k < d and k > ob:
            return self.signal(candles, SignalAction.SELL, 1.0, "bearish stoch cross overbought",
                               k=k, d=d)
        return self.signal(candles, SignalAction.HOLD, reason="no stoch signal")


class WilliamsRStrategy(Strategy):
    """Williams %R extremes. The scale is inverted (-100..0), so careful."""

    name = "williams_r"
    description = "%R below -80 is oversold (buy), above -20 is overbought (sell)"
    indicators = (("williams", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "oversold": -80, "overbought": -20}

    def prepare(self, candles):
        self._w = williams_r(candles, self.param("period", 14))

    def on_bar(self, i, candles):
        w = self._w[i]
        if w is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if w <= self.param("oversold", -80):
            return self.signal(candles, SignalAction.BUY, 1.0, "%R oversold", williams=w)
        if w >= self.param("overbought", -20):
            return self.signal(candles, SignalAction.SELL, 1.0, "%R overbought", williams=w)
        return self.signal(candles, SignalAction.HOLD, reason="%R neutral")


class CciStrategy(Strategy):
    """CCI crossing zero, with the +/-100 bands used as conviction."""

    name = "cci"
    description = "CCI crossing zero with strength scaled by the +/-100 bands"
    indicators = (("cci", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 20, "band": 100.0}

    def prepare(self, candles):
        self._c = cci(candles, self.param("period", 20))

    def on_bar(self, i, candles):
        c = self._c[i]
        if c is None or i < 1 or self._c[i - 1] is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        prev = self._c[i - 1]
        band = self.param("band", 100.0)
        if prev < 0 <= c:
            strength = min(1.0, (c / band) + 0.5)
            return self.signal(candles, SignalAction.BUY, strength, "CCI crossed above zero",
                               cci=c)
        if prev > 0 >= c:
            strength = min(1.0, (-c / band) + 0.5)
            return self.signal(candles, SignalAction.SELL, strength, "CCI crossed below zero",
                               cci=c)
        return self.signal(candles, SignalAction.HOLD, reason="CCI no cross")


class RsiAtrBreakout(Strategy):
    """RSI momentum entry with ATR-scaled conviction (a hybrid, deliberately)."""

    name = "rsi_atr_breakout"
    description = "RSI above 50 with expanding ATR buys; contracting ATR sells"
    indicators = (("rsi", 14), ("atr", 14))

    @classmethod
    def default_params(cls) -> dict:
        return {"rsi_period": 14, "atr_period": 14, "rsi_mid": 50}

    def prepare(self, candles):
        self._r = _rsi([c.close for c in candles], self.param("rsi_period", 14))
        self._a = atr(candles, self.param("atr_period", 14))

    def on_bar(self, i, candles):
        r, a = self._r[i], self._a[i]
        if r is None or a is None or i < 1 or self._a[i - 1] is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        mid = self.param("rsi_mid", 50)
        vol_expanding = a > self._a[i - 1]
        if r > mid and vol_expanding:
            return self.signal(candles, SignalAction.BUY, 0.8, "RSI bullish + volatility expanding",
                               rsi=r, atr=a)
        if r < mid and not vol_expanding:
            return self.signal(candles, SignalAction.SELL, 0.8, "RSI bearish + volatility contracting",
                               rsi=r, atr=a)
        return self.signal(candles, SignalAction.HOLD, reason="no confirmation")
