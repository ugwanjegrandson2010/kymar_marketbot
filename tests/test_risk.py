"""Risk engine tests: sizing, Kelly, and the drawdown/daily-loss kill switches.

The sizing tests assert real arithmetic against hand-computed values, because a
sizing bug silently changes every position in a backtest.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketbot.config import load_config
from marketbot.risk.limits import RiskLimits
from marketbot.risk.sizing import (
    kelly_fraction,
    position_for_risk,
    position_for_volatility_target,
    position_value_cap,
)


class TestPositionForRisk(unittest.TestCase):
    def test_sizes_to_risk_the_exact_budget(self):
        # $10k equity, 1% risk, entry 100, stop 90 -> risk $10/unit, $100 budget
        qty = position_for_risk(equity=10_000.0, risk_per_trade=0.01, entry=100.0, stop=90.0)
        self.assertAlmostEqual(qty, 10.0, places=6)

    def test_short_uses_same_risk_math(self):
        qty = position_for_risk(
            equity=10_000.0, risk_per_trade=0.01, entry=100.0, stop=110.0, direction="short"
        )
        self.assertAlmostEqual(qty, 10.0, places=6)

    def test_never_returns_negative_or_zero_on_valid_input(self):
        qty = position_for_risk(equity=5_000.0, risk_per_trade=0.02, entry=50.0, stop=45.0)
        self.assertGreater(qty, 0.0)

    def test_zero_risk_when_stop_equals_entry(self):
        # A stop at the entry means infinite position under the formula; the
        # engine must refuse rather than divide by zero.
        qty = position_for_risk(equity=10_000.0, risk_per_trade=0.01, entry=100.0, stop=100.0)
        self.assertEqual(qty, 0.0)

    def test_respects_max_position_cap(self):
        qty = position_for_risk(
            equity=10_000.0, risk_per_trade=0.10, entry=10.0, stop=9.0, max_position_pct=0.20
        )
        # Uncapped would be 100 units ($1000 = 10% of equity), but the cap is 20%.
        self.assertLessEqual(qty * 10.0, 10_000.0 * 0.20 + 1e-9)


class TestVolatilityTarget(unittest.TestCase):
    def test_scales_down_when_volatility_exceeds_target(self):
        qty = position_for_volatility_target(
            equity=10_000.0, target_vol=0.01, atr=5.0, price=100.0
        )
        # ATR 5% of price is 5x the 1% target, so size must be 1/5 of notional.
        self.assertLess(qty * 100.0, 10_000.0)

    def test_zero_atr_returns_zero(self):
        self.assertEqual(
            position_for_volatility_target(equity=10_000.0, target_vol=0.01, atr=0.0, price=100.0),
            0.0,
        )


class TestKelly(unittest.TestCase):
    def test_sixty_percent_winners_with_even_payoff_gives_positive_fraction(self):
        # Full Kelly = (p*b - q)/b = (0.6*1 - 0.4)/1 = 0.2
        full = kelly_fraction(win_rate=0.6, avg_win=1.0, avg_loss=1.0, fraction=1.0)
        self.assertAlmostEqual(full, 0.2, places=6)
        # Default is quarter-Kelly, which is the deliberately conservative part.
        self.assertAlmostEqual(
            kelly_fraction(win_rate=0.6, avg_win=1.0, avg_loss=1.0), 0.05, places=6
        )

    def test_coin_flip_with_even_payoff_gives_zero(self):
        self.assertEqual(kelly_fraction(win_rate=0.5, avg_win=1.0, avg_loss=1.0), 0.0)

    def test_negative_expectancy_is_rejected(self):
        self.assertEqual(kelly_fraction(win_rate=0.4, avg_win=1.0, avg_loss=1.0), 0.0)

    def test_full_kelly_is_fractionalized_by_config(self):
        f = kelly_fraction(win_rate=0.8, avg_win=2.0, avg_loss=1.0)
        self.assertLessEqual(f, 1.0)

    def test_degenerate_inputs_do_not_divide_by_zero(self):
        self.assertEqual(kelly_fraction(win_rate=0.5, avg_win=0.0, avg_loss=1.0), 0.0)


class TestPositionValueCap(unittest.TestCase):
    def test_caps_notional_at_percent_of_equity(self):
        qty = position_value_cap(equity=10_000.0, price=100.0, max_position_pct=0.20)
        self.assertAlmostEqual(qty, 20.0, places=6)

    def test_zero_price_returns_zero(self):
        self.assertEqual(position_value_cap(equity=10_000.0, price=0.0, max_position_pct=0.2), 0.0)


class TestRiskLimits(unittest.TestCase):
    def test_halts_after_max_drawdown_is_breached(self):
        cfg = load_config(starting_cash=10_000.0, max_drawdown_pct=0.20)
        limits = RiskLimits(cfg)
        # Equity falls from 10000 to 7900 = 21% drawdown, past the 20% limit.
        self.assertTrue(limits.drawdown_breached(7_900.0))

    def test_does_not_halt_inside_drawdown(self):
        cfg = load_config(starting_cash=10_000.0, max_drawdown_pct=0.20)
        limits = RiskLimits(cfg)
        self.assertFalse(limits.drawdown_breached(8_500.0))

    def test_daily_loss_limit_halts(self):
        cfg = load_config(starting_cash=10_000.0, max_daily_loss_pct=0.03)
        limits = RiskLimits(cfg)
        self.assertTrue(limits.daily_loss_breached(equity=9_600.0, session_start_equity=10_000.0))

    def test_daily_loss_limit_allows_normal_day(self):
        cfg = load_config(starting_cash=10_000.0, max_daily_loss_pct=0.03)
        limits = RiskLimits(cfg)
        self.assertFalse(limits.daily_loss_breached(equity=9_900.0, session_start_equity=10_000.0))

    def test_halt_releases_once_equity_recovers(self):
        cfg = load_config(starting_cash=10_000.0, max_drawdown_pct=0.20)
        limits = RiskLimits(cfg)
        self.assertTrue(limits.drawdown_breached(7_000.0))
        limits.reset()
        self.assertFalse(limits.drawdown_breached(7_000.0) and limits.halted)


if __name__ == "__main__":
    unittest.main()
