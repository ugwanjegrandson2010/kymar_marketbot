"""Trend-following strategies.

The shared idea: identify the direction of the dominant move and trade with it.
Every strategy here declares its lookback through `indicators` so the engine can
skip the warm-up bars instead of trading on uninitialized values.
"""

from __future__ import annotations

from ..indicators import atr as _atr
from ..indicators.core import adx, ema, sma
from ..indicators.core import supertrend as _supertrend
from ..indicators.core import donchian
from ..indicators.core import ichimoku
from ..model import SignalAction
from .base import Strategy


class SmaCrossover(Strategy):
    """Classic dual moving average: buy when fast crosses above slow."""

    name = "sma_crossover"
    description = "Fast SMA crossing above slow SMA"
    indicators = (("fast", 10), ("slow", 30))

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 10, "slow": 30}

    def prepare(self, candles):
        cl = [c.close for c in candles]
        self._fast = sma(cl, self.param("fast", 10))
        self._slow = sma(cl, self.param("slow", 30))

    def on_bar(self, i, candles):
        f, s = self._fast, self._slow
        if i < 1 or f[i] is None or s[i] is None or f[i - 1] is None or s[i - 1] is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        golden = f[i - 1] <= s[i - 1] and f[i] > s[i]
        death = f[i - 1] >= s[i - 1] and f[i] < s[i]
        if golden:
            return self.signal(candles, SignalAction.BUY, 1.0, "golden cross",
                               fast=f[i], slow=s[i])
        if death:
            return self.signal(candles, SignalAction.SELL, 1.0, "death cross",
                               fast=f[i], slow=s[i])
        return self.signal(candles, SignalAction.HOLD, reason="no cross")


class EmaTrend(Strategy):
    """Single EMA regime filter: long above, short below."""

    name = "ema_trend"
    description = "Price above EMA is long, below is short"
    indicators = (("ema", 50),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 50}

    def prepare(self, candles):
        self._e = ema([c.close for c in candles], self.param("period", 50))

    def on_bar(self, i, candles):
        e = self._e[i]
        if e is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        if price > e:
            return self.signal(candles, SignalAction.BUY, 1.0, "above EMA", ema=e)
        return self.signal(candles, SignalAction.SELL, 1.0, "below EMA", ema=e)


class MacdStrategy(Strategy):
    """MACD line crossing its signal line."""

    name = "macd"
    description = "MACD crossing its 9-period signal line"
    indicators = (("macd", 26),)

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 12, "slow": 26, "signal": 9}

    def prepare(self, candles):
        from ..indicators.core import macd

        cl = [c.close for c in candles]
        self._line, self._sig, self._hist = macd(
            cl, self.param("fast", 12), self.param("slow", 26), self.param("signal", 9)
        )

    def on_bar(self, i, candles):
        l, s, h = self._line, self._sig, self._hist
        if i < 1 or None in (l[i], s[i], l[i - 1], s[i - 1]):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if l[i - 1] <= s[i - 1] and l[i] > s[i]:
            return self.signal(candles, SignalAction.BUY, 1.0, "MACD bullish cross",
                               macd=l[i], signal=s[i])
        if l[i - 1] >= s[i - 1] and l[i] < s[i]:
            return self.signal(candles, SignalAction.SELL, 1.0, "MACD bearish cross",
                               macd=l[i], signal=s[i])
        return self.signal(candles, SignalAction.HOLD, reason="no MACD cross")


class AdxTrend(Strategy):
    """ADX-filtered +DI/-DI trend. Only trades when the trend is actually strong."""

    name = "adx_trend"
    description = "Trade +DI/-DI direction only when ADX confirms trend strength"
    indicators = (("adx", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "threshold": 20.0}

    def prepare(self, candles):
        self._adx, self._plus, self._minus = adx(candles, self.param("period", 14))

    def on_bar(self, i, candles):
        a, p, m = self._adx[i], self._plus[i], self._minus[i]
        if a is None or p is None or m is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if a < self.param("threshold", 20.0):
            return self.signal(candles, SignalAction.HOLD, 0.3, "ADX below threshold",
                               adx=a)
        if p > m:
            return self.signal(candles, SignalAction.BUY, min(1.0, a / 50.0),
                               "+DI above -DI", adx=a, plus_di=p, minus_di=m)
        return self.signal(candles, SignalAction.SELL, min(1.0, a / 50.0),
                           "-DI above +DI", adx=a, plus_di=p, minus_di=m)


class SupertrendStrategy(Strategy):
    """Supertrend flip: the trailing ATR band that reverses with price."""

    name = "supertrend"
    description = "Supertrend direction flip"
    indicators = (("atr", 10),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 10, "multiplier": 3.0}

    def prepare(self, candles):
        self._st = _supertrend(candles, self.param("period", 10), self.param("multiplier", 3.0))

    def on_bar(self, i, candles):
        st = self._st[i]
        if st is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        above = price > st
        if i > 0 and self._st[i - 1] is not None:
            was_above = candles[i - 1].close > self._st[i - 1]
            if above and not was_above:
                return self.signal(candles, SignalAction.BUY, 1.0, "supertrend bullish flip",
                                   supertrend=st)
            if not above and was_above:
                return self.signal(candles, SignalAction.SELL, 1.0, "supertrend bearish flip",
                                   supertrend=st)
        return self.signal(candles, SignalAction.HOLD, reason="supertrend unchanged")


class DonchianBreakout(Strategy):
    """Donchian channel breakout: the mechanic behind the Turtle system."""

    name = "donchian_breakout"
    description = "Break above N-bar high goes long, below N-bar low goes short"
    indicators = (("donchian", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"entry": 20, "exit": 10}

    def prepare(self, candles):
        self._entry_ch = donchian(candles, self.param("entry", 20))
        self._exit_ch = donchian(candles, self.param("exit", 10))

    def on_bar(self, i, candles):
        # Prior-bar channel: the current bar's own high/low must not define the
        # level being broken, or the breakout can never trigger.
        eu = self._entry_ch["upper"][i - 1] if i > 0 else None
        el = self._entry_ch["lower"][i - 1] if i > 0 else None
        xu = self._exit_ch["upper"][i - 1] if i > 0 else None
        xl = self._exit_ch["lower"][i - 1] if i > 0 else None
        if None in (eu, el, xu, xl):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        if price > eu:
            return self.signal(candles, SignalAction.BUY, 1.0, "above entry channel",
                               channel_high=eu)
        if price < el:
            return self.signal(candles, SignalAction.SELL, 1.0, "below entry channel",
                               channel_low=el)
        # Exit channel only produces a close, not a reversal.
        if price < xl or price > xu:
            return self.signal(candles, SignalAction.CLOSE, 1.0, "exit channel breached")
        return self.signal(candles, SignalAction.HOLD, reason="inside channel")


class IchimokuStrategy(Strategy):
    """Ichimoku cloud: trend requires price on the correct side of the kumo."""

    name = "ichimoku"
    description = "Tenkan/Kijun cross confirmed by price versus the cloud"
    indicators = (("tenkan", 9), ("kijun", 26), ("senkou_b", 52))

    @classmethod
    def default_params(cls) -> dict:
        return {"tenkan": 9, "kijun": 26, "senkou_b": 52}

    def prepare(self, candles):
        self._ic = ichimoku(
            candles, self.param("tenkan", 9), self.param("kijun", 26), self.param("senkou_b", 52)
        )

    def on_bar(self, i, candles):
        t, k = self._ic["tenkan"][i], self._ic["kijun"][i]
        sa, sb = self._ic["senkou_a"][i], self._ic["senkou_b"][i]
        if None in (t, k, sa, sb):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        cloud_top, cloud_bottom = max(sa, sb), min(sa, sb)
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom
        pt, pk = self._ic["tenkan"][i - 1], self._ic["kijun"][i - 1]
        if None not in (pt, pk) and pt <= pk and t > k and above_cloud:
            return self.signal(candles, SignalAction.BUY, 1.0, "bullish TK cross above cloud",
                               tenkan=t, kijun=k)
        if None not in (pt, pk) and pt >= pk and t < k and below_cloud:
            return self.signal(candles, SignalAction.SELL, 1.0, "bearish TK cross below cloud",
                               tenkan=t, kijun=k)
        return self.signal(candles, SignalAction.HOLD, reason="no TK cross or in cloud")
