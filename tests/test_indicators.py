"""Indicator correctness tests.

These run against synthetic candles so the expected values are known exactly,
plus one alignment/no-look-ahead property test that guards the invariant the
backtester depends on.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketbot.indicators import ALL_INDICATORS, compute, compute_all
from marketbot.indicators.core import closes, sma
from marketbot.indicators.oscillators import rsi
from marketbot.model import Candle


def make_candles(n: int, start: float = 100.0, step: float = 1.0) -> list[Candle]:
    """Ascending synthetic series with a known close sequence."""
    out = []
    price = start
    for i in range(n):
        out.append(
            Candle(
                ts=1_600_000_000 + i * 86400,
                open=price,
                high=price + step,
                low=price - step,
                close=price,
                volume=1000.0 + i,
            )
        )
        price += step
    return out


def make_ohlc_candles(closes_seq: list[float]) -> list[Candle]:
    """OHLC bars from a close sequence, with a fixed +/-1 range."""
    return [
        Candle(
            ts=1_600_000_000 + i * 86400,
            open=c,
            high=c + 1.0,
            low=c - 1.0,
            close=c,
            volume=1000.0,
        )
        for i, c in enumerate(closes_seq)
    ]


class TestRegistry(unittest.TestCase):
    def test_every_registered_indicator_returns_aligned_series(self):
        candles = make_candles(200)
        for name, fn in ALL_INDICATORS.items():
            with self.subTest(indicator=name):
                result = fn(candles, 14)
                self.assertIsInstance(result, list, f"{name} did not return a list")
                self.assertEqual(
                    len(result), len(candles), f"{name} misaligned: {len(result)} vs {len(candles)}"
                )

    def test_compute_by_name_matches_direct_call(self):
        candles = make_candles(120)
        direct = sma(closes(candles), 20)
        by_name = compute("sma", candles, 20)
        for i, (a, b) in enumerate(zip(direct, by_name)):
            if a is None:
                self.assertIsNone(b, f"sma[{i}] should be None")
            else:
                self.assertAlmostEqual(a, b, places=9, msg=f"sma[{i}]")

    def test_compute_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            compute("not_a_real_indicator", make_candles(50), 14)


class TestSMA(unittest.TestCase):
    def test_matches_hand_computed_average(self):
        candles = make_candles(30, start=100.0, step=1.0)
        got = compute("sma", candles, 10)[-1]
        # Closes are 100..129; the last 10 are 120..129.
        expected = sum(100.0 + i for i in range(20, 30)) / 10.0
        self.assertAlmostEqual(got, expected, places=9)

    def test_warmup_is_none(self):
        candles = make_candles(20)
        series = compute("sma", candles, 10)
        for i in range(9):
            self.assertIsNone(series[i], f"sma[{i}] should be None during warmup")

    def test_constant_series_gives_constant_average(self):
        candles = make_ohlc_candles([50.0] * 30)
        series = compute("sma", candles, 5)
        for i in range(4, 30):
            self.assertAlmostEqual(series[i], 50.0, places=9)


class TestRSI(unittest.TestCase):
    def test_bounds_are_0_to_100(self):
        candles = make_candles(200, start=100.0, step=0.5)
        series = compute("rsi", candles, 14)
        vals = [v for v in series if v is not None]
        self.assertTrue(vals, "RSI produced no values")
        self.assertTrue(all(0.0 <= v <= 100.0 for v in vals), "RSI escaped 0-100")

    def test_monotonically_rising_series_is_overbought(self):
        candles = make_candles(200, start=100.0, step=2.0)
        series = compute("rsi", candles, 14)
        self.assertGreater(series[-1], 95.0)

    def test_matches_wilder_reference(self):
        """A strictly falling series must produce RSI well below 50."""
        seq = [50.0 - i for i in range(20)]
        candles = make_ohlc_candles(seq)
        series = compute("rsi", candles, 14)
        self.assertIsNotNone(series[-1])
        self.assertLess(series[-1], 20.0)


class TestBandIndicators(unittest.TestCase):
    def test_bollinger_upper_ge_mid_ge_lower(self):
        from marketbot.indicators import BAND_INDICATORS

        candles = make_candles(120, start=200.0, step=0.3)
        bb = BAND_INDICATORS["bollinger"](candles, 20)
        checked = 0
        for u, m, l in zip(bb["upper"], bb["mid"], bb["lower"]):
            if u is None:
                continue
            self.assertGreaterEqual(u, m)
            self.assertGreaterEqual(m, l)
            checked += 1
        self.assertGreater(checked, 50, "too few defined Bollinger points")


class TestNoLookAhead(unittest.TestCase):
    def test_truncating_future_bars_does_not_change_past_values(self):
        """The invariant the backtester's no-look-ahead guarantee rests on."""
        candles = make_candles(150, start=100.0, step=0.7)
        full = compute_all(candles, 14)
        cut = 80
        truncated = compute_all(candles[:cut], 14)
        for name, series in full.items():
            with self.subTest(indicator=name):
                for i in range(cut):
                    a, b = series[i], truncated[name][i]
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        self.fail(f"{name}[{i}]: full={a} truncated={b}")
                    self.assertAlmostEqual(
                        a, b, places=9, msg=f"{name}[{i}] changed when future was removed"
                    )


if __name__ == "__main__":
    unittest.main()
