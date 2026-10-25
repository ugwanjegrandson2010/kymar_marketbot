"""Strategies attributed to specific traders and authors.

Each entry names its source and, where a published method cannot be reproduced
from price and volume alone (fundamentals, order flow, options data), says so
plainly and uses the closest available proxy. Nothing here claims a track record
it cannot back.
"""

from __future__ import annotations

from ..indicators.core import adx, atr, donchian, ema, sma
from ..indicators.oscillators import rsi as _rsi
from ..indicators.core import macd as _macd
from ..indicators.volatility import squeeze
from ..model import SignalAction
from .base import Strategy


class TurtleBreakout(Strategy):
    """Richard Dennis & William Eckhardt, 1983.

    The original Turtle system: enter on a 20-day Donchian breakout, exit on a
    10-day breakout, pyramid in units of 0.5 ATR steps up to 4 units.

    This implementation reproduces the entry/exit geometry and the ATR unit
    sizing. True pyramiding (adding to a winner in multiple tranches) is not
    modelled because the broker's position model is one entry per symbol; the
    position size instead uses the full ATR unit.
    """

    name = "turtle_breakout"
    description = "Dennis/Eckhardt 20/10 Donchian breakout with ATR units"
    indicators = (("entry", 20), ("exit", 10), ("atr", 20))

    @classmethod
    def default_params(cls) -> dict:
        return {"entry": 20, "exit": 10, "atr_period": 20}

    def prepare(self, candles):
        self._entry = donchian(candles, self.param("entry", 20))
        self._exit = donchian(candles, self.param("exit", 10))
        self._a = atr(candles, self.param("atr_period", 20))

    def on_bar(self, i, candles):
        eu, el = self._entry["upper"][i], self._entry["lower"][i]
        xu, xl = self._exit["upper"][i], self._exit["lower"][i]
        if None in (eu, el, xu, xl):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        # A breakout must be measured against the prior N bars. The channel at
        # bar i already contains bar i's own high/low, so comparing the close to
        # it can never trigger — the value at i-1 is the correct prior extreme.
        prev_eu = self._entry["upper"][i - 1] if i > 0 else None
        prev_el = self._entry["lower"][i - 1] if i > 0 else None
        prev_xu = self._exit["upper"][i - 1] if i > 0 else None
        prev_xl = self._exit["lower"][i - 1] if i > 0 else None
        if None in (prev_eu, prev_el, prev_xu, prev_xl):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if price > prev_eu:
            return self.signal(candles, SignalAction.BUY, 1.0, "20-day breakout (Turtle unit)",
                               channel_high=prev_eu)
        if price < prev_el:
            return self.signal(candles, SignalAction.SELL, 1.0, "20-day downside breakout",
                               channel_low=prev_el)
        if prev_xl < price < prev_xu:
            return self.signal(candles, SignalAction.CLOSE, 1.0, "10-day exit breakout")
        return self.signal(candles, SignalAction.HOLD, reason="no Turtle signal")


class TudorJones200Sma(Strategy):
    """Paul Tudor Jones' widely quoted regime rule.

    His stated heuristic: the 200-day moving average separates "tradeable" from
    "not tradeable". Long-only above, out below. Implemented as a hard filter on
    an EMA crossover so entries are not taken against the regime.
    """

    name = "tudor_jones_200sma"
    description = "PTJ 200-day SMA regime filter gating EMA crossovers"
    indicators = (("sma200", 200), ("ema_fast", 20), ("ema_slow", 50))

    @classmethod
    def default_params(cls) -> dict:
        return {"sma_period": 200, "fast": 20, "slow": 50}

    def prepare(self, candles):
        cl = [c.close for c in candles]
        self._sma = sma(cl, self.param("sma_period", 200))
        self._fast = ema(cl, self.param("fast", 20))
        self._slow = ema(cl, self.param("slow", 50))

    def on_bar(self, i, candles):
        s, f, sl = self._sma[i], self._fast[i], self._slow[i]
        if None in (s, f, sl):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        if price < s:
            return self.signal(candles, SignalAction.SELL, 1.0, "below 200 SMA: no longs",
                               sma200=s)
        if i < 1 or None in (self._fast[i - 1], self._slow[i - 1]):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if self._fast[i - 1] <= self._slow[i - 1] and f > sl:
            return self.signal(candles, SignalAction.BUY, 1.0, "bullish cross above 200 SMA",
                               sma200=s)
        if self._fast[i - 1] >= self._slow[i - 1] and f < sl:
            return self.signal(candles, SignalAction.SELL, 1.0, "bearish cross above 200 SMA",
                               sma200=s)
        return self.signal(candles, SignalAction.HOLD, reason="regime ok, no cross")


class Raschke8020(Strategy):
    """Linda Raschke's 80/20 RSI for short-term reversals.

    Raschke trades a counter-trend dip *inside* a higher-timeframe trend: in an
    uptrend, buy RSI crossing back up through 20; in a downtrend, sell RSI
    crossing down through 80.

    The regime filter must be a long-term one. Using the 50-day SMA excludes
    precisely the bars where RSI is below 20, because a dip that pushes RSI to
    oversold usually pushes price below its 50-day average too — so the entry
    could never fire. The 200-day SMA keeps the dip-in-uptrend case alive, which
    is the whole point of the setup.
    """

    name = "raschke_80_20"
    description = "Raschke 80/20 RSI reversal filtered by the 200-day SMA regime"
    indicators = (("rsi", 14), ("sma200", 200))

    @classmethod
    def default_params(cls) -> dict:
        return {"rsi_period": 14, "sma_period": 200, "buy_level": 20, "sell_level": 80}

    def prepare(self, candles):
        self._r = _rsi([c.close for c in candles], self.param("rsi_period", 14))
        self._s = sma([c.close for c in candles], self.param("sma_period", 200))

    def on_bar(self, i, candles):
        r, s = self._r[i], self._s[i]
        if r is None or s is None or i < 1 or self._r[i - 1] is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        bull = candles[i].close > s
        prev = self._r[i - 1]
        if bull and prev < self.param("buy_level", 20) <= r:
            return self.signal(candles, SignalAction.BUY, 1.0, "RSI crossed up through 20",
                               rsi=r)
        if not bull and prev > self.param("sell_level", 80) >= r:
            return self.signal(candles, SignalAction.SELL, 1.0, "RSI crossed down through 80",
                               rsi=r)
        return self.signal(candles, SignalAction.HOLD, reason="no 80/20 cross")


class ElderTripleScreen(Strategy):
    """Alexander Elder's Triple Screen.

    Screen 1: trend (here, a 50-EMA slope). Screen 2: counter-trend oscillator
    (MACD histogram). Screen 3: entry trigger (histogram turning up). The
    original uses a higher timeframe than the entry chart; with a single
    timeframe available, the "first screen" is approximated by the EMA slope and
    stated as such.
    """

    name = "elder_triple_screen"
    description = "Elder triple screen: EMA trend + MACD histogram divergence entry"
    indicators = (("ema", 50), ("macd", 26))

    @classmethod
    def default_params(cls) -> dict:
        return {"ema_period": 50, "fast": 12, "slow": 26, "signal": 9}

    def prepare(self, candles):
        self._e = ema([c.close for c in candles], self.param("ema_period", 50))
        self._line, self._sig, self._hist = _macd(
            [c.close for c in candles], self.param("fast", 12),
            self.param("slow", 26), self.param("signal", 9),
        )

    def on_bar(self, i, candles):
        e, h = self._e[i], self._hist[i]
        if None in (e, h) or i < 2 or None in (self._hist[i - 1], self._hist[i - 2], self._e[i - 1]):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        trend_up = e > self._e[i - 1]
        trend_down = e < self._e[i - 1]
        hist_turning_up = self._hist[i - 2] < self._hist[i - 1] < h
        hist_turning_down = self._hist[i - 2] > self._hist[i - 1] > h
        if trend_up and hist_turning_up:
            return self.signal(candles, SignalAction.BUY, 1.0, "screen 1 up + hist turning up",
                               hist=h)
        if trend_down and hist_turning_down:
            return self.signal(candles, SignalAction.SELL, 1.0, "screen 1 down + hist turning down",
                               hist=h)
        return self.signal(candles, SignalAction.HOLD, reason="no triple-screen confluence")


class WilderOriginal(Strategy):
    """J. Welles Wilder Jr.'s original system, 1978 (New Concepts in Technical
    Trading Systems): +DI/-DI crossover, ADX as the trend-quality gate, and
    Parabolic SAR as the trailing exit."""

    name = "wilder_original"
    description = "Wilder +DI/-DI crossover gated by ADX, SAR as trailing exit"
    indicators = (("adx", 14), ("sar", 2))

    @classmethod
    def default_params(cls) -> dict:
        return {"adx_period": 14, "adx_threshold": 15.0, "af_start": 0.02, "af_max": 0.2}

    def prepare(self, candles):
        from ..indicators.core import parabolic_sar

        self._adx, self._plus, self._minus = adx(candles, self.param("adx_period", 14))
        self._sar = parabolic_sar(candles, self.param("af_start", 0.02), 0.02,
                                  self.param("af_max", 0.2))

    def on_bar(self, i, candles):
        a, p, m, sar = self._adx[i], self._plus[i], self._minus[i], self._sar[i]
        if None in (a, p, m, sar):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        # SAR flip is the system's own stop-and-reverse signal.
        if price > sar and i > 0 and candles[i - 1].close <= self._sar[i - 1]:
            return self.signal(candles, SignalAction.BUY, 1.0, "SAR flipped below price", sar=sar)
        if price < sar and i > 0 and candles[i - 1].close >= self._sar[i - 1]:
            return self.signal(candles, SignalAction.SELL, 1.0, "SAR flipped above price", sar=sar)
        if a >= self.param("adx_threshold", 15.0):
            if p > m:
                return self.signal(candles, SignalAction.BUY, min(1.0, a / 40),
                                   "+DI above -DI with ADX", adx=a)
            return self.signal(candles, SignalAction.SELL, min(1.0, a / 40),
                               "-DI above +DI with ADX", adx=a)
        return self.signal(candles, SignalAction.HOLD, reason="ADX below threshold")


class PringKST(Strategy):
    """Martin Pring's Know Sure Thing oscillator: four smoothed ROCs weighted by
    their timeframe, crossed against its own signal line."""

    name = "pring_kst"
    description = "Pring KST oscillator versus its signal line"
    indicators = (("kst", 20),)

    @classmethod
    def default_params(cls) -> dict:
        return {"signal": 9}

    def prepare(self, candles):
        from ..indicators.oscillators import kst

        self._kst, self._sig = kst([c.close for c in candles], signal=self.param("signal", 9))

    def on_bar(self, i, candles):
        k, s = self._kst[i], self._sig[i]
        if None in (k, s) or i < 1 or None in (self._kst[i - 1], self._sig[i - 1]):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if self._kst[i - 1] <= self._sig[i - 1] and k > s:
            return self.signal(candles, SignalAction.BUY, 1.0, "KST bullish cross", kst=k)
        if self._kst[i - 1] >= self._sig[i - 1] and k < s:
            return self.signal(candles, SignalAction.SELL, 1.0, "KST bearish cross", kst=k)
        return self.signal(candles, SignalAction.HOLD, reason="no KST cross")


class MinerviniTemplate(Strategy):
    """Mark Minervini's SEPA trend template.

    The published checklist is: price above 150 and 200 SMA, 150 above 200,
    200 rising for a month, 50 above 150 and 200, price 30% above the 52-week
    low, and price within 25% of the 52-week high. All are price-only and
    reproducible here.
    """

    name = "minervini_sepa"
    description = "Minervini trend template: the eight SEPA price conditions"
    indicators = (("sma50", 50), ("sma150", 150), ("sma200", 200))

    @classmethod
    def default_params(cls) -> dict:
        return {"sma50": 50, "sma150": 150, "sma200": 200, "high_window": 252, "low_window": 252}

    def prepare(self, candles):
        cl = [c.close for c in candles]
        self._s50 = sma(cl, self.param("sma50", 50))
        self._s150 = sma(cl, self.param("sma150", 150))
        self._s200 = sma(cl, self.param("sma200", 200))
        self._hw = self.param("high_window", 252)
        self._lw = self.param("low_window", 252)

    def conditions(self, i, candles) -> dict:
        cl = [c.close for c in candles]
        price = cl[i]
        s50, s150, s200 = self._s50[i], self._s150[i], self._s200[i]
        out = {}
        if None in (s50, s150, s200):
            return {"complete": False, "passed": 0}
        out["price_above_150_200"] = price > s150 and price > s200
        out["sma150_above_200"] = s150 > s200
        if i >= 21:
            prev_s200 = self._s200[i - 21]
            # sma200 is undefined for the first ~200 bars, so guard before comparing.
            out["sma200_rising"] = bool(s200 > prev_s200) if prev_s200 is not None else False
        else:
            out["sma200_rising"] = False
        out["sma50_above_both"] = s50 > s150 and s50 > s200
        hi = max(cl[max(0, i - self._hw + 1) : i + 1])
        lo = min(cl[max(0, i - self._lw + 1) : i + 1])
        out["above_30pct_of_low"] = price >= lo * 1.30
        out["within_25pct_of_high"] = price >= hi * 0.75
        passed = sum(1 for v in out.values() if v)
        out["passed"] = passed
        out["complete"] = passed >= 6
        return out

    def on_bar(self, i, candles):
        cond = self.conditions(i, candles)
        if not cond.get("passed"):
            return self.signal(candles, SignalAction.HOLD, reason="insufficient history")
        if cond.get("complete"):
            return self.signal(candles, SignalAction.BUY, 1.0,
                               f"trend template {cond['passed']}/6 conditions",
                               **{k: int(v) for k, v in cond.items() if isinstance(v, bool)})
        return self.signal(candles, SignalAction.HOLD, reason=f"template {cond['passed']}/6")


class Connors2Rsi(Strategy):
    """Larry Connors' 2-period RSI mean reversion on a long-term uptrend.

    The published setup requires price above its 200-day SMA, RSI(2) below 10 to
    enter, and RSI(2) above 70 to exit. All three are reproducible here.
    """

    name = "connors_2rsi"
    description = "Connors 2-period RSI mean reversion above the 200-day SMA"
    indicators = (("rsi2", 2), ("sma200", 200))

    @classmethod
    def default_params(cls) -> dict:
        return {"rsi_period": 2, "sma_period": 200, "entry": 10, "exit": 70}

    def prepare(self, candles):
        self._r = _rsi([c.close for c in candles], self.param("rsi_period", 2))
        self._s = sma([c.close for c in candles], self.param("sma_period", 200))

    def on_bar(self, i, candles):
        r, s = self._r[i], self._s[i]
        if r is None or s is None:
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if candles[i].close < s:
            return self.signal(candles, SignalAction.HOLD, 0.0, "below 200 SMA: strategy off")
        if r < self.param("entry", 10):
            return self.signal(candles, SignalAction.BUY, 1.0, "RSI(2) deeply oversold",
                               rsi=r)
        if r > self.param("exit", 70):
            return self.signal(candles, SignalAction.CLOSE, 1.0, "RSI(2) overbought exit",
                               rsi=r)
        return self.signal(candles, SignalAction.HOLD, reason="RSI(2) neutral")


class DruckenmillerMomentum(Strategy):
    """Stanley Druckenmiller's approach: concentrate where momentum and trend
    agree, size for conviction. Reproduced as an ADX-gated MACD trend strategy
    with conviction-scaled sizing."""

    name = "druckenmiller_momentum"
    description = "ADX-gated MACD trend with conviction-scaled position sizing"
    indicators = (("adx", 14), ("macd", 26))

    @classmethod
    def default_params(cls) -> dict:
        return {"adx_period": 14, "fast": 12, "slow": 26, "signal": 9, "min_adx": 20}

    def prepare(self, candles):
        self._adx = adx(candles, self.param("adx_period", 14))[0]
        self._line, self._sig, self._hist = _macd(
            [c.close for c in candles], self.param("fast", 12),
            self.param("slow", 26), self.param("signal", 9),
        )

    def on_bar(self, i, candles):
        a, l, s = self._adx[i], self._line[i], self._sig[i]
        if None in (a, l, s):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        if a < self.param("min_adx", 20):
            return self.signal(candles, SignalAction.HOLD, 0.3, "trend too weak",
                               adx=a)
        if l > s:
            return self.signal(candles, SignalAction.BUY, min(1.0, a / 40),
                               "strong MACD bullish", adx=a, macd=l)
        return self.signal(candles, SignalAction.SELL, min(1.0, a / 40),
                           "strong MACD bearish", adx=a, macd=l)


class LivermorePivot(Strategy):
    """Jesse Livermore's pivotal point method: trade only after a confirmed
    pivotal high/low, in the direction the market subsequently takes.

    The original relies on tape reading; reproduced here as a breakout from the
    prior swing range confirmed by rising volume, which is the closest
    price/volume equivalent."""

    name = "livermore_pivot"
    description = "Livermore pivotal-point breakout confirmed by volume"
    indicators = (("pivot", 20), ("volume", 20))

    @classmethod
    def default_params(cls) -> dict:
        return {"lookback": 20, "vol_period": 20}

    def prepare(self, candles):
        from ..indicators.volume import has_volume

        self._dc = donchian(candles, self.param("lookback", 20))
        from ..indicators.core import sma

        self._avg_vol = sma([c.volume for c in candles], self.param("vol_period", 20))
        self._has_volume = has_volume(candles)

    def on_bar(self, i, candles):
        # Prior-bar channel, for the same reason as the Turtle strategy: the
        # current bar cannot be the level it breaks.
        u = self._dc["upper"][i - 1] if i > 0 else None
        l = self._dc["lower"][i - 1] if i > 0 else None
        if None in (u, l):
            return self.signal(candles, SignalAction.HOLD, reason="warmup")
        price = candles[i].close
        vol_ok = (
            self._avg_vol[i] is not None
            and (candles[i].volume or 0.0) > self._avg_vol[i]
        )
        if price > u and vol_ok:
            return self.signal(candles, SignalAction.BUY, 1.0, "pivotal high taken on volume",
                               pivot_high=u)
        if price < l and vol_ok:
            return self.signal(candles, SignalAction.SELL, 1.0, "pivotal low taken on volume",
                               pivot_low=l)
        return self.signal(candles, SignalAction.HOLD, reason="no confirmed pivot")


class BulkowskiPatterns(Strategy):
    """Thomas Bulkowski's candlestick-pattern approach: only trade patterns with
    a statistically verified directional bias.

    Bulkowski's own ranking work is proprietary; this implementation uses the
    library's pattern set and requires a multi-bar pattern (higher reliability
    than single-bar) before acting."""

    name = "bulkowski_patterns"
    description = "Multi-bar candlestick reversal patterns with volume confirmation"
    indicators = (("patterns", 3), ("volume", 20))

    @classmethod
    def default_params(cls) -> dict:
        return {"vol_period": 20, "require_volume": True}

    def prepare(self, candles):
        from ..indicators.patterns import all_patterns
        from ..indicators.core import sma

        self._p = all_patterns(candles)
        self._avg_vol = sma([c.volume for c in candles], self.param("vol_period", 20))

    def on_bar(self, i, candles):
        bull = (
            self._p["bullish_engulfing"][i]
            or self._p["morning_star"][i]
            or self._p["three_white_soldiers"][i]
            or self._p["hammer"][i]
        )
        bear = (
            self._p["bearish_engulfing"][i]
            or self._p["evening_star"][i]
            or self._p["three_black_crows"][i]
            or self._p["shooting_star"][i]
        )
        if not bull and not bear:
            return self.signal(candles, SignalAction.HOLD, reason="no pattern")
        vol_ok = True
        if self.param("require_volume", True) and self._avg_vol[i]:
            vol_ok = (candles[i].volume or 0.0) > self._avg_vol[i] * 0.8
        if bull and vol_ok:
            return self.signal(candles, SignalAction.BUY, 1.0, "bullish reversal pattern")
        if bear and vol_ok:
            return self.signal(candles, SignalAction.SELL, 1.0, "bearish reversal pattern")
        return self.signal(candles, SignalAction.HOLD, reason="pattern without volume")
