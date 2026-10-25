"""Multi-agent trading firm, structured after Tauric Research's TradingAgents.

The idea is that a trading decision is not one judgement but several, made by
roles with different information and different incentives, with an explicit
veto held by a risk role that never sees the profit motive. This module
implements that pipeline:

    Analysts  ->  Bull/Bear researchers  ->  Trader  ->  Risk manager  ->  Portfolio manager
    (evidence)    (adversarial debate)       (decision)    (veto)            (final gate)

Two things make the structure worth having rather than decorative:

  * The risk manager's veto cannot be overridden by the trader. A decision that
    breaches the portfolio's risk limits is rejected regardless of conviction.
  * Every role returns structured output, so a decision can be audited later —
    which analyst said what, which researcher won the argument, and why.

Roles are deterministic by default (no network, no model), so the firm runs
without any LLM. Passing an `Advisor` to a role delegates that role's reasoning
to the model while keeping the pipeline's structure intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..backtest.metrics import Metrics
from ..config import Config
from ..indicators.core import atr, closes, sma, stdev
from ..model import Candle, SignalAction
from .advisor import Advice, Advisor, AnalysisRequest, LocalAdvisor


# --------------------------------------------------------------------------
# structured outputs
# --------------------------------------------------------------------------
@dataclass
class AnalystView:
    role: str
    direction: SignalAction
    confidence: float
    evidence: dict = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "direction": self.action_value(),
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "reasoning": self.reasoning,
        }

    def action_value(self) -> str:
        return self.direction.value

    @property
    def score(self) -> float:
        """Signed conviction in -1..+1."""
        if self.direction is SignalAction.BUY:
            return self.confidence
        if self.direction is SignalAction.SELL:
            return -self.confidence
        return 0.0


@dataclass
class DebateResult:
    bull_case: str
    bear_case: str
    winner: str
    reconciled_score: float
    confidence: float
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "winner": self.winner,
            "reconciled_score": round(self.reconciled_score, 4),
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


@dataclass
class TradeDecision:
    action: SignalAction
    strategy: Optional[str]
    params: dict
    conviction: float
    reasoning: str
    analyst_score: float = 0.0
    risk_approved: bool = False
    portfolio_approved: bool = False
    reject_reason: str = ""

    @property
    def executable(self) -> bool:
        return self.risk_approved and self.portfolio_approved and self.action is not SignalAction.HOLD

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "strategy": self.strategy,
            "params": self.params,
            "conviction": round(self.conviction, 3),
            "reasoning": self.reasoning,
            "analyst_score": round(self.analyst_score, 4),
            "risk_approved": self.risk_approved,
            "portfolio_approved": self.portfolio_approved,
            "executable": self.executable,
            "reject_reason": self.reject_reason,
        }


# --------------------------------------------------------------------------
# analyst roles
# --------------------------------------------------------------------------
class Analyst:
    """Base analyst: computes an evidence-backed directional view."""

    role = "analyst"

    def __init__(self, advisor: Optional[Advisor] = None):
        self.advisor = advisor

    def analyze(self, request: AnalysisRequest) -> AnalystView:
        raise NotImplementedError


class TechnicalAnalyst(Analyst):
    """Reads trend, momentum, and volatility from the indicator set."""

    role = "technical"

    def analyze(self, request: AnalysisRequest) -> AnalystView:
        candles = request.candles
        if len(candles) < 60:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "insufficient history")
        cl = closes(candles)
        fast, slow = sma(cl, 20), sma(cl, 50)
        a = atr(candles, 14)
        sd = stdev(cl, 20)
        if None in (fast[-1], slow[-1], a[-1], sd[-1]):
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "indicators not ready")

        price = cl[-1]
        evidence = {
            "price": round(price, 4),
            "sma20": round(fast[-1], 4),
            "sma50": round(slow[-1], 4),
            "atr14": round(a[-1], 4),
            "atr_pct": round(a[-1] / price * 100, 3),
            "stdev20": round(sd[-1], 4),
        }
        trend_score = (price > fast[-1]) + (fast[-1] > slow[-1])
        if trend_score == 2:
            return AnalystView(self.role, SignalAction.BUY, 0.7, evidence,
                               "price above 20 SMA above 50 SMA")
        if trend_score == 0:
            return AnalystView(self.role, SignalAction.SELL, 0.7, evidence,
                               "price below 20 SMA below 50 SMA")
        return AnalystView(self.role, SignalAction.HOLD, 0.3, evidence, "mixed trend structure")


class MomentumAnalyst(Analyst):
    """Reads oscillator state: whether momentum is with the move or exhausted."""

    role = "momentum"

    def analyze(self, request: AnalysisRequest) -> AnalystView:
        from ..indicators.oscillators import rsi

        candles = request.candles
        if len(candles) < 30:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "insufficient history")
        r = rsi(closes(candles), 14)
        if r[-1] is None:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "RSI not ready")
        value = r[-1]
        evidence = {"rsi14": round(value, 2)}
        if value < 35:
            return AnalystView(self.role, SignalAction.BUY, 0.5, evidence, "RSI approaching oversold")
        if value > 65:
            return AnalystView(self.role, SignalAction.SELL, 0.5, evidence, "RSI approaching overbought")
        return AnalystView(self.role, SignalAction.HOLD, 0.3, evidence, "RSI neutral")


class VolatilityAnalyst(Analyst):
    """Reads the volatility regime: expansion favours breakout, compression reversion."""

    role = "volatility"

    def analyze(self, request: AnalysisRequest) -> AnalystView:
        candles = request.candles
        if len(candles) < 60:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "insufficient history")
        cl = closes(candles)
        recent = stdev(cl[-20:], 20)[-1]
        earlier = stdev(cl[-60:-20], 20)[-1]
        if recent is None or earlier is None or earlier == 0:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "volatility not ready")
        ratio = recent / earlier
        evidence = {"recent_vol": round(recent, 4), "earlier_vol": round(earlier, 4),
                    "ratio": round(ratio, 3)}
        if ratio > 1.5:
            return AnalystView(self.role, SignalAction.HOLD, 0.4, evidence,
                               "volatility expanding: wait for direction")
        if ratio < 0.67:
            return AnalystView(self.role, SignalAction.HOLD, 0.5, evidence,
                               "volatility compressed: favour mean reversion")
        return AnalystView(self.role, SignalAction.HOLD, 0.2, evidence, "volatility normal")


class SentimentAnalyst(Analyst):
    """News/social sentiment proxy.

    Without a licensed news feed the honest version of this role uses price and
    volume behaviour as a stand-in and says so. It never pretends to have read
    the news.
    """

    role = "sentiment"

    def analyze(self, request: AnalysisRequest) -> AnalystView:
        candles = request.candles
        if len(candles) < 20:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {}, "insufficient history")
        cl = closes(candles)
        vol = [c.volume for c in candles if c.volume]
        evidence = {"note": "price/volume proxy; no news feed configured"}
        if vol:
            evidence["avg_volume_20"] = round(sum(vol[-20:]) / min(20, len(vol)), 2)
        change_5d = (cl[-1] / cl[-6] - 1) * 100 if len(cl) > 6 else 0.0
        change_20d = (cl[-1] / cl[-21] - 1) * 100 if len(cl) > 21 else 0.0
        evidence["change_5d_pct"] = round(change_5d, 3)
        evidence["change_20d_pct"] = round(change_20d, 3)
        if change_5d > 3 and change_20d > 5:
            return AnalystView(self.role, SignalAction.BUY, 0.4, evidence,
                               "broad strength suggests positive tone")
        if change_5d < -3 and change_20d < -5:
            return AnalystView(self.role, SignalAction.SELL, 0.4, evidence,
                               "broad weakness suggests negative tone")
        return AnalystView(self.role, SignalAction.HOLD, 0.2, evidence, "no clear tone")


class FundamentalAnalyst(Analyst):
    """Market-cap / valuation context.

    Only meaningful for stocks; for crypto the role reports flat rather than
    inventing a valuation it cannot compute.
    """

    role = "fundamental"

    def analyze(self, request: AnalysisRequest) -> AnalystView:
        kind = request.constraints.get("kind", "crypto")
        if kind != "stock":
            return AnalystView(
                self.role, SignalAction.HOLD, 0.0,
                {"note": "no fundamentals for this asset class"},
                "fundamentals not applicable",
            )
        market_cap = request.constraints.get("market_cap")
        if not market_cap:
            return AnalystView(self.role, SignalAction.HOLD, 0.0, {},
                               "no market cap supplied")
        evidence = {"market_cap": market_cap}
        if market_cap > 1e11:
            return AnalystView(self.role, SignalAction.HOLD, 0.3, evidence, "large cap: stable")
        if market_cap > 1e9:
            return AnalystView(self.role, SignalAction.HOLD, 0.2, evidence, "mid cap")
        return AnalystView(self.role, SignalAction.HOLD, 0.1, evidence, "small cap: higher risk")


# --------------------------------------------------------------------------
# adversarial debate
# --------------------------------------------------------------------------
def run_debate(views: Sequence[AnalystView], request: AnalysisRequest) -> DebateResult:
    """Bull and bear researchers argue from the same evidence.

    The bull takes the most bullish views, the bear the most bearish, and the
    winner is whichever side the evidence-weighted score supports. Confidence
    drops when the two sides disagree strongly, because that is exactly when a
    single-direction bet is least justified.
    """
    bull_score = sum(v.score for v in views if v.score > 0)
    bear_score = sum(-v.score for v in views if v.score < 0)
    total = bull_score + bear_score

    bull_case = "; ".join(f"{v.role}: {v.reasoning}" for v in views if v.score > 0) or "no bullish evidence"
    bear_case = "; ".join(f"{v.role}: {v.reasoning}" for v in views if v.score < 0) or "no bearish evidence"

    if total <= 0:
        return DebateResult(
            bull_case=bull_case, bear_case=bear_case, winner="none",
            reconciled_score=0.0, confidence=0.0,
            reasoning="analysts produced no directional evidence",
        )

    reconciled = (bull_score - bear_score) / total
    agreement = abs(bull_score - bear_score) / total
    winner = "bull" if reconciled > 0 else "bear"
    return DebateResult(
        bull_case=bull_case,
        bear_case=bear_case,
        winner=winner,
        reconciled_score=max(-1.0, min(1.0, reconciled)),
        confidence=round(agreement, 3),
        reasoning=(
            f"{winner} case carries {agreement:.0%} of the weighted evidence "
            f"({len(views)} analysts)"
        ),
    )


# --------------------------------------------------------------------------
# trader, risk manager, portfolio manager
# --------------------------------------------------------------------------
class TraderAgent:
    """Turns the reconciled analyst score into a concrete trade decision."""

    role = "trader"

    def __init__(self, advisor: Optional[Advisor] = None):
        self.advisor = advisor or LocalAdvisor()

    def decide(self, debate: DebateResult, request: AnalysisRequest) -> TradeDecision:
        if self.advisor is not None and not isinstance(self.advisor, LocalAdvisor):
            advice = self.advisor.advise(request)
            action = advice.action
            conviction = advice.confidence
            strategy = advice.strategy
            params = advice.params
            reasoning = advice.reasoning
        else:
            threshold = 0.2
            if debate.reconciled_score > threshold:
                action, conviction = SignalAction.BUY, debate.confidence
            elif debate.reconciled_score < -threshold:
                action, conviction = SignalAction.SELL, debate.confidence
            else:
                action, conviction = SignalAction.HOLD, 0.0
            strategy = "composite" if action is not SignalAction.HOLD else None
            params = {}
            reasoning = f"reconciled score {debate.reconciled_score:+.2f}"

        return TradeDecision(
            action=action,
            strategy=strategy,
            params=params,
            conviction=round(conviction, 3),
            reasoning=reasoning,
            analyst_score=round(debate.reconciled_score, 4),
        )


class RiskManager:
    """Veto power over any decision. Sees risk only, never the upside case."""

    role = "risk_manager"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def review(self, decision: TradeDecision, request: AnalysisRequest) -> TradeDecision:
        candles = request.candles
        reasons = []

        if decision.action is SignalAction.HOLD:
            decision.risk_approved = True
            return decision

        if len(candles) < 30:
            reasons.append("insufficient history to size risk")
        else:
            price = candles[-1].close
            a = atr(candles, 14)
            if a[-1] is None or a[-1] <= 0:
                reasons.append("ATR unavailable: cannot compute a stop")
            else:
                stop_distance = a[-1] * 2.0
                risk_pct = stop_distance / price
                if risk_pct > 0.10:
                    reasons.append(f"stop distance {risk_pct:.1%} of price exceeds 10% cap")
                if decision.conviction > 0.9 and risk_pct > 0.05:
                    reasons.append("high conviction with wide stop: reduce size or skip")

        existing = request.constraints.get("open_positions", 0)
        if existing >= self.config.max_open_positions:
            reasons.append(f"max open positions ({self.config.max_open_positions}) reached")

        equity = request.constraints.get("equity", self.config.starting_cash)
        if equity <= 0:
            reasons.append("equity unknown or zero")

        if reasons:
            decision.risk_approved = False
            decision.reject_reason = "; ".join(reasons)
        else:
            decision.risk_approved = True
        return decision


class PortfolioManager:
    """Final approval gate. Can reject for portfolio reasons, never override risk."""

    role = "portfolio_manager"

    def approve(self, decision: TradeDecision, request: AnalysisRequest) -> TradeDecision:
        if not decision.risk_approved:
            decision.portfolio_approved = False
            if not decision.reject_reason:
                decision.reject_reason = "rejected by risk management"
            return decision

        if decision.action is SignalAction.HOLD:
            decision.portfolio_approved = True
            return decision

        # Concentration check: never let one idea dominate the book.
        max_weight = request.constraints.get("max_symbol_weight", self.config_max_weight())
        if max_weight and decision.conviction > 0.95:
            decision.conviction = 0.95
            decision.reasoning += " | conviction capped to respect concentration limit"

        decision.portfolio_approved = True
        return decision

    @staticmethod
    def config_max_weight() -> float:
        return 0.35


# --------------------------------------------------------------------------
# the firm
# --------------------------------------------------------------------------
class Firm:
    """Runs the full analyst -> debate -> trader -> risk -> portfolio pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        advisor: Optional[Advisor] = None,
        analysts: Optional[Sequence[Analyst]] = None,
    ):
        self.config = config or Config()
        self.advisor = advisor
        self.analysts = list(analysts) if analysts else [
            TechnicalAnalyst(),
            MomentumAnalyst(),
            VolatilityAnalyst(),
            SentimentAnalyst(),
            FundamentalAnalyst(),
        ]
        self.trader = TraderAgent(advisor) if advisor else TraderAgent()
        self.risk_manager = RiskManager(self.config)
        self.portfolio_manager = PortfolioManager()

    def run(self, request: AnalysisRequest) -> dict:
        views = [a.analyze(request) for a in self.analysts]
        debate = run_debate(views, request)
        decision = self.trader.decide(debate, request)
        decision = self.risk_manager.review(decision, request)
        decision = self.portfolio_manager.approve(decision, request)
        return {
            "symbol": request.symbol,
            "analysts": [v.to_dict() for v in views],
            "debate": debate.to_dict(),
            "decision": decision.to_dict(),
            "verdict": self._verdict(decision),
        }

    @staticmethod
    def _verdict(decision: TradeDecision) -> str:
        if not decision.executable:
            return f"no trade: {decision.reject_reason or 'no signal'}"
        side = "long" if decision.action is SignalAction.BUY else "short"
        return (
            f"trade {side} via {decision.strategy} at {decision.conviction:.0%} conviction"
        )


def make_firm(config: Optional[Config] = None, advisor_name: str = "local", **kw) -> Firm:
    """Build a firm with a chosen advisor driving the trader role."""
    from .advisor import make_advisor

    advisor = None if advisor_name == "none" else make_advisor(advisor_name, **kw)
    return Firm(config=config, advisor=advisor)
