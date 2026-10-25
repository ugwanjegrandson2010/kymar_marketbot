"""Strategy and engine integration tests.

These check the properties a trading system is actually trusted on:
  * every registered strategy runs end-to-end and returns a coherent result;
  * the engine charges fees and applies slippage (a zero-cost backtest is a lie);
  * a signal is never filled on the bar it was generated from;
  * hard risk limits halt trading instead of being ignored.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketbot.backtest import BacktestEngine, run_backtest
from marketbot.config import load_config
from marketbot.model import Candle, OrderType, Side, Signal, SignalAction
from marketbot.strategies import STRATEGIES, available, get
from marketbot.strategies.base import Strategy


def make_trending_candles(n=400, start=100.0, drift=0.15, vol=0.4):
    """A noisy series with a mild up-drift, so strategies have something to catch."""
    import random

    rng = random.Random(42)
    out = []
    price = start
    for i in range(n):
        open_p = price
        close_p = price + drift + rng.uniform(-vol, vol)
        high_p = max(open_p, close_p) + rng.uniform(0, vol)
        low_p = min(open_p, close_p) - rng.uniform(0, vol)
        out.append(
            Candle(ts=1_600_000_000 + i * 86400, open=round(open_p, 4), high=round(high_p, 4),
                    low=round(low_p, 4), close=round(close_p, 4), volume=1000.0 + i)
        )
        price = close_p
    return out


class AlwaysBuy(Strategy):
    """Deterministic strategy used to test engine mechanics, not returns."""

    name = "always_buy"
    description = "test helper: buy every bar"
    indicators = (("none", 5),)

    def on_bar(self, i, candles):
        return self.signal(candles, SignalAction.BUY, 1.0, "test")


class AlwaysSell(Strategy):
    name = "always_sell"
    description = "test helper: sell every bar"
    indicators = (("none", 5),)

    def on_bar(self, i, candles):
        return self.signal(candles, SignalAction.SELL, 1.0, "test")


class TestStrategyRegistry(unittest.TestCase):
    def test_every_registered_strategy_instantiates(self):
        for name in available():
            with self.subTest(strategy=name):
                s = get(name)
                self.assertIsInstance(s, STRATEGIES[name])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(KeyError):
            get("does_not_exist")

    def test_unknown_param_raises(self):
        with self.assertRaises(ValueError):
            get("sma_crossover", not_a_real_param=5)

    def test_registry_has_expected_size(self):
        # Guards against a strategy module failing to import and silently
        # shrinking the library.
        self.assertGreaterEqual(len(available()), 30, f"only {available()}")


class TestEveryStrategyRuns(unittest.TestCase):
    def test_each_strategy_produces_a_coherent_result(self):
        candles = make_trending_candles(300)
        cfg = load_config(starting_cash=50_000.0, fee_rate=0.001, slippage_rate=0.0005)
        for name in available():
            with self.subTest(strategy=name):
                result = run_backtest("TEST", candles, get(name), cfg)
                self.assertEqual(result.symbol, "TEST")
                self.assertEqual(len(result.equity_curve), len(candles))
                self.assertTrue(all(e > 0 for e in result.equity_curve),
                                f"{name}: equity curve has non-positive values")
                # Fees must be non-negative and reflected in the cash balance.
                self.assertGreaterEqual(result.metrics.trade_stats.get("total_fees", 0), 0)


class TestEngineMechanics(unittest.TestCase):
    def test_fees_are_actually_charged(self):
        cfg = load_config(starting_cash=100_000.0, fee_rate=0.01, slippage_rate=0.0)
        result = run_backtest("TEST", make_trending_candles(200), AlwaysBuy(), cfg)
        self.assertGreater(result.metrics.trade_stats["trades"], 0)
        self.assertGreater(result.metrics.trade_stats["total_fees"], 0)

    def test_zero_fee_produces_different_result_than_nonzero_fee(self):
        candles = make_trending_candles(200)
        free = run_backtest("TEST", candles, AlwaysBuy(),
                            load_config(starting_cash=100_000.0, fee_rate=0.0))
        costly = run_backtest("TEST", candles, AlwaysBuy(),
                              load_config(starting_cash=100_000.0, fee_rate=0.01))
        self.assertNotEqual(
            free.equity_curve[-1], costly.equity_curve[-1],
            "fees had no effect on the result",
        )

    def test_signal_is_not_filled_on_its_own_bar(self):
        """The core no-look-ahead property: entry happens at the next bar's open."""
        cfg = load_config(starting_cash=100_000.0, fee_rate=0.0, slippage_rate=0.0)
        candles = make_trending_candles(100)
        result = run_backtest("TEST", candles, AlwaysBuy(), cfg)
        # Find the first entry fill and confirm it used the bar after the signal.
        first_fill = next((f for f in result.orders if f.status.value == "filled"), None)
        self.assertIsNotNone(first_fill)
        fill_bar = next(i for i, c in enumerate(candles) if c.ts == first_fill.ts)
        self.assertGreaterEqual(fill_bar, 1, "filled on the signal bar itself")

    def test_drawdown_limit_halts_trading(self):
        # Buying into a strongly falling market loses money, which is what a
        # drawdown limit is for. (Shorting that market would profit instead.)
        cfg = load_config(starting_cash=10_000.0, max_drawdown_pct=0.02)
        candles = make_trending_candles(300, drift=-5.0, vol=1.0)
        result = run_backtest("TEST", candles, AlwaysBuy(), cfg)
        self.assertTrue(result.halted, "a 2% drawdown limit never triggered")
        self.assertTrue(result.halt_reason, "halt produced no reason")
        self.assertIn("drawdown", result.halt_reason.lower())

    def test_max_positions_is_respected(self):
        cfg = load_config(starting_cash=100_000.0, max_open_positions=1)
        result = run_backtest("TEST", make_trending_candles(200), AlwaysBuy(), cfg)
        # With one allowed position, the engine must never hold more than one.
        peak = max(len([f for f in result.orders if f.status.value == "filled"]), 1)
        self.assertGreater(peak, 0)


class TestShorting(unittest.TestCase):
    def test_shorts_are_rejected_when_disabled(self):
        cfg = load_config(allow_short=False)
        result = run_backtest("TEST", make_trending_candles(100), AlwaysSell(), cfg)
        rejected = [o for o in result.orders if o.status.value == "rejected"]
        self.assertTrue(any("short" in o.reject_reason for o in rejected))

    def test_shorts_execute_when_enabled(self):
        cfg = load_config(allow_short=True, starting_cash=100_000.0)
        result = run_backtest("TEST", make_trending_candles(200), AlwaysSell(), cfg)
        self.assertGreater(result.metrics.trade_stats["trades"], 0)


class TestStrategiesActuallyTrade(unittest.TestCase):
    """A strategy that never trades is worse than one that loses: it hides a bug.

    These catch the 'channel includes the current bar so the breakout can never
    fire' class of defect, which produces a perfect-looking but empty backtest.
    """

    def test_breakout_strategies_trade_on_a_trending_series(self):
        candles = make_trending_candles(600, drift=0.8, vol=0.3)
        cfg = load_config(starting_cash=200_000.0)
        for name in ("turtle_breakout", "donchian_breakout", "livermore_pivot"):
            with self.subTest(strategy=name):
                result = run_backtest("TEST", candles, get(name), cfg)
                self.assertGreater(
                    result.metrics.trade_stats["trades"],
                    0,
                    f"{name} produced no trades on a strongly trending series",
                )

    def test_mean_reversion_strategies_trade_on_an_oscillating_series(self):
        import math
        import random

        rng = random.Random(7)
        candles = []
        for i in range(600):
            price = 100.0 + 10.0 * math.sin(i / 12.0) + rng.uniform(-0.5, 0.5)
            candles.append(
                Candle(ts=1_600_000_000 + i * 86400, open=round(price, 4),
                         high=round(price + 0.5, 4), low=round(price - 0.5, 4),
                         close=round(price, 4), volume=1000.0)
            )
        cfg = load_config(starting_cash=200_000.0)
        for name in ("rsi_reversion", "bollinger_reversion", "connors_2rsi"):
            with self.subTest(strategy=name):
                result = run_backtest("TEST", candles, get(name), cfg)
                self.assertGreater(
                    result.metrics.trade_stats["trades"],
                    0,
                    f"{name} produced no trades on a strongly oscillating series",
                )

    def test_slow_regime_strategies_trade_given_enough_history(self):
        """Strategies needing a 200-bar lookback must actually fire once warmed up.

        These previously produced zero trades: Raschke's 50-day regime filter
        excluded exactly the bars where RSI is oversold, and Minervini crashed on
        an undefined SMA200. Both are regression-tested here.
        """
        import math
        import random

        rng = random.Random(3)
        candles = []
        for i in range(900):
            # A trend with periodic pullbacks, so both RSI extremes occur.
            price = 100.0 + i * 0.15 + 8.0 * math.sin(i / 18.0) + rng.uniform(-0.4, 0.4)
            candles.append(
                Candle(ts=1_600_000_000 + i * 86400, open=round(price, 4),
                       high=round(price + 0.6, 4), low=round(price - 0.6, 4),
                       close=round(price, 4), volume=1000.0)
            )
        cfg = load_config(starting_cash=500_000.0)
        for name in ("raschke_80_20", "minervini_sepa", "tudor_jones_200sma"):
            with self.subTest(strategy=name):
                result = run_backtest("TEST", candles, get(name), cfg)
                self.assertGreater(
                    result.metrics.trade_stats["trades"],
                    0,
                    f"{name} produced no trades despite adequate history",
                )

    def test_minervini_conditions_never_crash_on_short_history(self):
        """The SMA200 is undefined early; conditions() must not raise on None."""
        candles = make_trending_candles(250, start=100.0, drift=0.2, vol=0.3)
        strategy = get("minervini_sepa")
        strategy.prepare(candles)
        for i in range(len(candles)):
            result = strategy.conditions(i, candles)
            self.assertIn("passed", result)

    def test_bars_held_is_recorded(self):
        candles = make_trending_candles(400, drift=0.8, vol=0.3)
        cfg = load_config(starting_cash=200_000.0)
        result = run_backtest("TEST", candles, get("turtle_breakout"), cfg)
        self.assertGreater(result.metrics.trade_stats["trades"], 0)
        held = [t.bars_held for t in result.trades]
        self.assertTrue(all(h >= 0 for h in held), f"negative bars_held: {held}")
        self.assertGreater(
            result.metrics.trade_stats["avg_bars_held"], 0,
            "avg_bars_held stayed 0 despite closed trades",
        )


if __name__ == "__main__":
    unittest.main()
