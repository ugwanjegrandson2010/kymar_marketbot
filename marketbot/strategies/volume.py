"""Volume and money-flow strategies.

Volume confirms or denies a price move. The distinction these strategies encode:
rising price on rising volume is conviction; rising price on falling volume is
distribution and is treated as a warning rather than a signal.
"""

from __future__ import annotations

from ..indicators.volume import has_volume, mfi, obv, vwap_session
from ..model import SignalAction
from .base import Strategy


class VwapMeanReversion(Strategy):
    """Fade extended moves away from session VWAP."""

    name = "vwap_reversion"
    description = "Buy when price is below session VWAP by a threshold, sell above"
    indicators = (("vwap", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"threshold_pct": 1.0}

    def prepare(self, candles):
        self._v = vwap_session(candles)
        self._has_volume = has_volume(candles)

    def on_bar(self, i, candles):
        v = self._v[i]
        if v is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        dev = (price - v) / v * 100.0
        th = self.param("threshold_pct", 1.0)
        if dev < -th:
            return self.signal(candles, SignalAction.BUY, min(1.0, -dev / (th * 3)),
                               "below VWAP", vwap=v, deviation=round(dev, 3))
        if dev > th:
            return self.signal(candles, SignalAction.SELL, min(1.0, dev / (th * 3)),
                               "above VWAP", vwap=v, deviation=round(dev, 3))
        return self.signal(candles, SignalAction.HOLD, reason="near VWAP")


class MfiStrategy(Strategy):
    """Money Flow Index: volume-weighted RSI extremes."""

    name = "mfi"
    description = "MFI below 20 buys, above 80 sells, exit through 50"
    indicators = (("mfi", 14),)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "oversold": 20, "overbought": 80, "exit": 50}

    def prepare(self, candles):
        self._m = mfi(candles, self.param("period", 14))
        self._has_volume = has_volume(candles)

    def on_bar(self, i, candles):
        m = self._m[i]
        if m is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        os_, ob = self.param("oversold", 20), self.param("overbought", 80)
        if m < os_:
            return self.signal(candles, SignalAction.BUY, 1.0, "MFI oversold", mfi=m)
        if m > ob:
            return self.signal(candles, SignalAction.SELL, 1.0, "MFI overbought", mfi=m)
        prev = self._m[i - 1] if i > 0 else None
        ex = self.param("exit", 50)
        if prev is not None and prev < ex <= m:
            return self.signal(candles, SignalAction.CLOSE, 1.0, "MFI recovered")
        return self.signal(candles, SignalAction.HOLD, reason="MFI neutral")


class ObvDivergence(Strategy):
    """OBV trend confirmation: trade price direction only when OBV agrees."""

    name = "obv_confirmation"
    description = "Buy price strength confirmed by OBV making a new high, and vice versa"
    indicators = (("obv", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"lookback": 20}

    def prepare(self, candles):
        self._o = obv(candles)
        self._lb = self.param("lookback", 20)

    def on_bar(self, i, candles):
        if i < self._lb:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        obv_now = self._o[i]
        if obv_now is None:
            return self.signal(candles, SignalAction.HOLD, reason="no obv")
        window = self._o[i - self._lb : i]
        price_window = [c.close for c in candles[i - self._lb : i]]
        price_new_high = price > max(price_window)
        price_new_low = price < min(price_window)
        obv_new_high = obv_now > max(window)
        obv_new_low = obv_now < min(window)

        if price_new_high and obv_new_high:
            return self.signal(candles, SignalAction.BUY, 1.0, "price and OBV new highs",
                               obv=obv_now)
        if price_new_low and obv_new_low:
            return self.signal(candles, SignalAction.SELL, 1.0, "price and OBV new lows",
                               obv=obv_now)
        # Divergence: price makes a new high but OBV does not confirm.
        if price_new_high and not obv_new_high:
            return self.signal(candles, SignalAction.SELL, 0.7, "bearish divergence",
                               obv=obv_now)
        if price_new_low and not obv_new_low:
            return self.signal(candles, SignalAction.BUY, 0.7, "bullish divergence",
                               obv=obv_now)
        return self.signal(candles, SignalAction.HOLD, reason="no OBV signal")


class VolumePriceTrend(Strategy):
    """Volume-weighted trend: buy rising price on above-average volume."""

    name = "volume_price_trend"
    description = "Rising closes on above-average volume buy; falling closes below-average sell"
    indicators = (("volume", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"vol_period": 20, "price_period": 5}

    def prepare(self, candles):
        from ..indicators.core import sma

        self._avg_vol = sma([c.volume for c in candles], self.param("vol_period", 20))
        self._has_volume = has_volume(candles)

    def on_bar(self, i, candles):
        av = self._avg_vol[i]
        if av is None or av <= 0:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        v = candles[i].volume or 0.0
        rising = candles[i].close > candles[i - 1].close if i > 0 else False
        falling = candles[i].close < candles[i - 1].close if i > 0 else False
        heavy = v > av * 1.5
        if rising and heavy:
            return self.signal(candles, SignalAction.BUY, min(1.0, v / (av * 3)),
                               "rising price on heavy volume", avg_volume=av, volume=v)
        if falling and heavy:
            return self.signal(candles, SignalAction.SELL, min(1.0, v / (av * 3)),
                               "falling price on heavy volume", avg_volume=av, volume=v)
        return self.signal(candles, SignalAction.HOLD, reason="no volume conviction")
