"""Tests for the AI analysis layer: advisors, debate, and the risk veto.

The most important assertion here is that the risk manager's veto cannot be
overridden — that property is what stops an enthusiastic model from trading
through a risk limit.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketbot.ai.advisor import Advice, AnalysisRequest, HermesAdvisor, LocalAdvisor, make_advisor
from marketbot.ai.multiagent import (
    DebateResult,
    Firm,
    FundamentalAnalyst,
    MomentumAnalyst,
    PortfolioManager,
    RiskManager,
    SentimentAnalyst,
    TechnicalAnalyst,
    TradeDecision,
    VolatilityAnalyst,
    run_debate,
)
from marketbot.config import load_config
from marketbot.model import Candle, SignalAction


def make_candles(n=200, drift=0.5, vol=0.3):
    import random

    rng = random.Random(11)
    out = []
    price = 100.0
    for i in range(n):
        o = price
        c = price + drift + rng.uniform(-vol, vol)
        out.append(Candle(ts=1_600_000_000 + i * 86400, open=round(o, 4),
                          high=round(max(o, c) + 0.2, 4), low=round(min(o, c) - 0.2, 4),
                          close=round(c, 4), volume=1000.0 + (i % 50) * 10))
        price = c
    return out


def make_request(candles=None, **kw):
    return AnalysisRequest(
        symbol=kw.get("symbol", "TEST-USD"),
        candles=candles if candles is not None else make_candles(),
        strategy_options=kw.get("strategy_options", ["composite", "sma_crossover"]),
        current_position=kw.get("current_position"),
        question=kw.get("question", "What should I do?"),
        constraints=kw.get("constraints", {}),
    )


class TestLocalAdvisor(unittest.TestCase):
    def test_holds_on_insufficient_history(self):
        advice = LocalAdvisor().advise(make_request(candles=make_candles(20)))
        self.assertIs(advice.action, SignalAction.HOLD)

    def test_picks_a_trend_strategy_in_an_uptrend(self):
        advice = LocalAdvisor().advise(make_request(candles=make_candles(200, drift=0.6)))
        self.assertIn(advice.action, (SignalAction.BUY, SignalAction.HOLD))
        if advice.action is SignalAction.BUY:
            self.assertIsNotNone(advice.strategy)

    def test_returns_structured_advice(self):
        advice = LocalAdvisor().advise(make_request())
        d = advice.to_dict()
        self.assertIn("action", d)
        self.assertIn("confidence", d)
        self.assertTrue(0.0 <= d["confidence"] <= 1.0)


class TestHermesAdvisor(unittest.TestCase):
    def test_rejects_invalid_action(self):
        advisor = HermesAdvisor()
        request = make_request()
        parsed = {"action": "YOLO", "strategy": "composite", "confidence": 0.9}
        advice = advisor._validate(parsed, request, "{}")
        self.assertIs(advice.action, SignalAction.HOLD)
        self.assertIn("invalid action", advice.reasoning)

    def test_rejects_strategy_outside_allowlist(self):
        advisor = HermesAdvisor()
        parsed = {"action": "buy", "strategy": "not_allowed", "confidence": 0.9}
        advice = advisor._validate(parsed, make_request(), "{}")
        self.assertIs(advice.action, SignalAction.HOLD)
        self.assertIn("not in the allowed list", advice.reasoning)

    def test_enforces_param_constraints_over_model_output(self):
        advisor = HermesAdvisor()
        parsed = {"action": "buy", "strategy": "composite",
                  "params": {"period": 999}, "confidence": 0.9}
        request = make_request(constraints={"allowed_params": {"period": [10, 20, 50]}})
        advice = advisor._validate(parsed, request, "{}")
        self.assertEqual(advice.params["period"], 10)

    def test_clamps_confidence(self):
        advisor = HermesAdvisor()
        parsed = {"action": "buy", "strategy": "composite", "confidence": 5.0}
        advice = advisor._validate(parsed, make_request(), "{}")
        self.assertLessEqual(advice.confidence, 1.0)

    def test_extracts_json_from_surrounding_prose(self):
        text = 'Here is my analysis.\n{"action": "hold", "confidence": 0.1}\nDone.'
        self.assertIsNotNone(HermesAdvisor._extract_json(text))

    def test_returns_hold_when_advisor_unavailable(self):
        advisor = HermesAdvisor(command="/nonexistent/binary", timeout=5)
        advice = advisor.advise(make_request())
        self.assertIs(advice.action, SignalAction.HOLD)


class TestMakeAdvisor(unittest.TestCase):
    def test_unknown_advisor_raises(self):
        with self.assertRaises(KeyError):
            make_advisor("does_not_exist")


class TestAnalysts(unittest.TestCase):
    def test_technical_analyst_direction_follows_trend(self):
        up = TechnicalAnalyst().analyze(make_request(candles=make_candles(200, drift=0.8)))
        down = TechnicalAnalyst().analyze(make_request(candles=make_candles(200, drift=-0.8)))
        self.assertIs(up.direction, SignalAction.BUY)
        self.assertIs(down.direction, SignalAction.SELL)

    def test_analysts_hold_on_short_history(self):
        req = make_request(candles=make_candles(10))
        for analyst in (TechnicalAnalyst(), MomentumAnalyst(), VolatilityAnalyst()):
            with self.subTest(analyst=analyst.role):
                self.assertIs(analyst.analyze(req).direction, SignalAction.HOLD)

    def test_fundamental_analyst_is_flat_for_crypto(self):
        view = FundamentalAnalyst().analyze(
            make_request(constraints={"kind": "crypto"})
        )
        self.assertIs(view.direction, SignalAction.HOLD)

    def test_sentiment_analyst_reports_proxy_limitation(self):
        view = SentimentAnalyst().analyze(make_request())
        self.assertIn("note", view.evidence)


class TestDebate(unittest.TestCase):
    def test_bull_wins_with_bullish_evidence(self):
        from marketbot.ai.multiagent import AnalystView

        views = [
            AnalystView("technical", SignalAction.BUY, 0.8),
            AnalystView("momentum", SignalAction.BUY, 0.5),
            AnalystView("volatility", SignalAction.HOLD, 0.2),
        ]
        result = run_debate(views, make_request())
        self.assertEqual(result.winner, "bull")
        self.assertGreater(result.reconciled_score, 0)

    def test_conflicting_evidence_lowers_confidence(self):
        from marketbot.ai.multiagent import AnalystView

        conflict = [
            AnalystView("technical", SignalAction.BUY, 0.8),
            AnalystView("momentum", SignalAction.SELL, 0.8),
        ]
        result = run_debate(conflict, make_request())
        self.assertEqual(result.reconciled_score, 0.0)
        self.assertEqual(result.confidence, 0.0)

    def test_no_evidence_yields_no_winner(self):
        from marketbot.ai.multiagent import AnalystView

        result = run_debate([AnalystView("a", SignalAction.HOLD, 0.5)], make_request())
        self.assertEqual(result.winner, "none")
        self.assertEqual(result.reconciled_score, 0.0)


class TestRiskVeto(unittest.TestCase):
    def setUp(self):
        self.config = load_config(max_open_positions=3)
        self.manager = RiskManager(self.config)
        self.pm = PortfolioManager()

    def test_rejects_when_history_is_insufficient(self):
        decision = TradeDecision(SignalAction.BUY, "composite", {}, 0.8, "test")
        reviewed = self.manager.review(decision, make_request(candles=make_candles(5)))
        self.assertFalse(reviewed.risk_approved)
        self.assertIn("insufficient history", reviewed.reject_reason)

    def test_rejects_when_max_positions_reached(self):
        decision = TradeDecision(SignalAction.BUY, "composite", {}, 0.5, "test")
        request = make_request(constraints={"open_positions": 3})
        reviewed = self.manager.review(decision, request)
        self.assertFalse(reviewed.risk_approved)
        self.assertIn("max open positions", reviewed.reject_reason)

    def test_veto_cannot_be_overridden_by_portfolio_manager(self):
        """The core guarantee: a risk rejection survives the final gate."""
        decision = TradeDecision(SignalAction.BUY, "composite", {}, 0.9, "very confident")
        request = make_request(candles=make_candles(5))
        reviewed = self.manager.review(decision, request)
        approved = self.pm.approve(reviewed, request)
        self.assertFalse(approved.risk_approved)
        self.assertFalse(approved.portfolio_approved)
        self.assertFalse(approved.executable)

    def test_approves_a_normal_trade(self):
        decision = TradeDecision(SignalAction.BUY, "composite", {}, 0.5, "test")
        reviewed = self.manager.review(decision, make_request())
        approved = self.pm.approve(reviewed, make_request())
        self.assertTrue(approved.executable)

    def test_hold_needs_no_approval(self):
        decision = TradeDecision(SignalAction.HOLD, None, {}, 0.0, "nothing to do")
        reviewed = self.manager.review(decision, make_request(candles=make_candles(2)))
        self.assertTrue(reviewed.risk_approved)
        self.assertFalse(reviewed.executable)


class TestFirm(unittest.TestCase):
    def test_runs_the_full_pipeline(self):
        firm = Firm(config=load_config())
        result = firm.run(make_request())
        self.assertIn("analysts", result)
        self.assertIn("debate", result)
        self.assertIn("decision", result)
        self.assertEqual(len(result["analysts"]), 5)

    def test_decision_is_structured(self):
        result = Firm(config=load_config()).run(make_request())
        d = result["decision"]
        self.assertIn("action", d)
        self.assertIn("risk_approved", d)
        self.assertIn("portfolio_approved", d)
        self.assertIn("executable", d)

    def test_verdict_is_plain_language(self):
        result = Firm(config=load_config()).run(make_request())
        self.assertIsInstance(result["verdict"], str)
        self.assertTrue(len(result["verdict"]) > 0)

    def test_make_firm_with_none_advisor(self):
        firm = Firm(config=load_config(), advisor=None)
        result = firm.run(make_request())
        self.assertIn("decision", result)


if __name__ == "__main__":
    unittest.main()
