"""Pluggable analysis layer: let an AI do the middle work.

The bot's mechanics (indicators, backtest, risk, execution) are deterministic.
Deciding *what to do with them* — which strategy, which parameters, whether a
signal is worth acting on, what a result means — is judgement, and that is what
this layer hands to a model.

Two adapters ship:
  * `LocalAdvisor` — deterministic rules, no network, always available. Good as
    a baseline and as the thing an AI's output is compared against.
  * `HermesAdvisor` — drives Hermes (this agent) through its CLI, passing a
    structured prompt and parsing a structured reply.

The advisor never places orders. It returns Advice, and the caller decides
whether to act, which keeps the risk limits outside the model's reach.
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..backtest.metrics import Metrics
from ..model import Candle, SignalAction


@dataclass
class Advice:
    action: SignalAction = SignalAction.HOLD
    strategy: Optional[str] = None
    params: dict = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    risk_note: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "strategy": self.strategy,
            "params": self.params,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "risk_note": self.risk_note,
        }


@dataclass
class AnalysisRequest:
    """Everything an advisor needs, in one structured object."""

    symbol: str
    candles: Sequence[Candle]
    strategy_options: Sequence[str]
    current_position: Optional[str] = None
    recent_metrics: Optional[Metrics] = None
    question: str = "What should I trade and how should I size it?"
    constraints: dict = field(default_factory=dict)

    def summary(self) -> dict:
        if not self.candles:
            return {"symbol": self.symbol, "bars": 0}
        last = self.candles[-1]
        window = self.candles[-min(50, len(self.candles)):]
        closes = [c.close for c in window]
        return {
            "symbol": self.symbol,
            "bars": len(self.candles),
            "last_ts": last.ts,
            "last_close": round(last.close, 6),
            "last_change_pct": round((closes[-1] / closes[0] - 1) * 100, 3),
            "recent_high": round(max(c.high for c in window), 6),
            "recent_low": round(min(c.low for c in window), 6),
            "avg_volume": round(sum(c.volume for c in window) / len(window), 2),
            "current_position": self.current_position,
            "strategies": list(self.strategy_options),
            "question": self.question,
            "constraints": self.constraints,
            "metrics": self.recent_metrics.to_dict() if self.recent_metrics else None,
        }


class Advisor(ABC):
    """Base class for analysis providers."""

    name = "base"

    @abstractmethod
    def advise(self, request: AnalysisRequest) -> Advice:
        raise NotImplementedError


class LocalAdvisor(Advisor):
    """Rule-based advisor: no model, no network, deterministic.

    Uses the composite indicator score and the recent trend to pick between a
    small set of strategies. This is the baseline an AI advisor should beat, and
    the fallback when no model is configured.
    """

    name = "local"

    def advise(self, request: AnalysisRequest) -> Advice:
        from ..indicators.core import closes, sma
        from ..strategies import available

        summary = request.summary()
        if summary["bars"] < 60:
            return Advice(
                action=SignalAction.HOLD,
                reasoning="insufficient history to make a call",
                confidence=0.0,
            )

        cl = closes(request.candles)
        fast = sma(cl, 20)
        slow = sma(cl, 50)
        if fast[-1] is None or slow[-1] is None:
            return Advice(action=SignalAction.HOLD, reasoning="indicators not ready")

        price = cl[-1]
        above = price > fast[-1] > slow[-1]
        below = price < fast[-1] < slow[-1]
        opts = list(request.strategy_options) or available()

        if above:
            for candidate in ("turtle_breakout", "adx_trend", "sma_crossover", "composite"):
                if candidate in opts:
                    return Advice(
                        action=SignalAction.BUY,
                        strategy=candidate,
                        params={},
                        confidence=0.6,
                        reasoning=(
                            f"uptrend structure (price > SMA20 > SMA50); "
                            f"selecting trend-following {candidate}"
                        ),
                        risk_note="size by risk_per_trade, not conviction",
                    )
        if below:
            for candidate in ("wilder_original", "tudor_jones_200sma", "composite"):
                if candidate in opts:
                    return Advice(
                        action=SignalAction.SELL,
                        strategy=candidate,
                        params={},
                        confidence=0.6,
                        reasoning=(
                            f"downtrend structure (price < SMA20 < SMA50); "
                            f"selecting {candidate}"
                        ),
                        risk_note="confirm regime before sizing",
                    )
        return Advice(
            action=SignalAction.HOLD,
            reasoning="no clear trend structure; staying flat",
            confidence=0.4,
        )


class HermesAdvisor(Advisor):
    """Drives the Hermes agent CLI to produce an analysis.

    The model receives a compact JSON brief and must reply with JSON matching the
    Advice schema. A reply that does not parse, or that violates the constraints,
    is rejected rather than acted on.
    """

    name = "hermes"

    def __init__(self, command: str = "hermes", timeout: int = 300):
        self.command = command
        self.timeout = timeout

    def build_prompt(self, request: AnalysisRequest) -> str:
        return (
            "You are the analysis layer of a trading bot. Decide what to do.\n"
            "Reply with ONLY a JSON object, no prose, matching this schema:\n"
            '{"action": "buy"|"sell"|"hold"|"close", "strategy": "<name or null>", '
            '"params": {}, "confidence": 0.0, "reasoning": "<one sentence>", '
            '"risk_note": "<one sentence>"}\n\n'
            f"Available strategies: {', '.join(request.strategy_options)}\n"
            f"Current position: {request.current_position or 'flat'}\n"
            f"Constraints: {json.dumps(request.constraints)}\n\n"
            f"Market brief:\n{json.dumps(request.summary(), indent=2)}\n"
        )

    def advise(self, request: AnalysisRequest) -> Advice:
        prompt = self.build_prompt(request)
        try:
            proc = subprocess.run(
                [self.command, "-p", prompt, "--no-tools"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return Advice(
                action=SignalAction.HOLD,
                reasoning=f"advisor unavailable: {e}",
                confidence=0.0,
                raw="",
            )
        raw = (proc.stdout or "").strip()
        if not raw:
            return Advice(action=SignalAction.HOLD, reasoning="advisor returned nothing",
                          confidence=0.0, raw=proc.stderr[:200])
        parsed = self._extract_json(raw)
        if parsed is None:
            return Advice(action=SignalAction.HOLD, reasoning="advisor reply not JSON",
                          confidence=0.0, raw=raw[:500])
        return self._validate(parsed, request, raw)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Find the first JSON object in the reply, tolerating prose around it."""
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None

    def _validate(self, parsed: dict, request: AnalysisRequest, raw: str) -> Advice:
        action_raw = str(parsed.get("action", "hold")).lower()
        if action_raw not in ("buy", "sell", "hold", "close"):
            return Advice(action=SignalAction.HOLD, reasoning=f"invalid action {action_raw!r}",
                          confidence=0.0, raw=raw[:300])
        strategy = parsed.get("strategy")
        if strategy and request.strategy_options and strategy not in request.strategy_options:
            return Advice(
                action=SignalAction.HOLD,
                reasoning=f"strategy {strategy!r} is not in the allowed list",
                confidence=0.0,
                raw=raw[:300],
            )
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            # A reply that cannot be parsed to a number is not acted on. Silently
            # defaulting the confidence to 0 would still leave a BUY signal
            # standing, which is exactly what must not happen.
            return Advice(
                action=SignalAction.HOLD,
                reasoning=f"confidence is not numeric: {parsed.get('confidence')!r}",
                confidence=0.0,
                raw=raw[:300],
            )
        params = parsed.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        # Constraints win over the model: never allow a param outside the bounds.
        for key, value in params.items():
            allowed = request.constraints.get("allowed_params", {}).get(key)
            if isinstance(allowed, list) and value not in allowed:
                params[key] = allowed[0]
        return Advice(
            action=SignalAction(action_raw),
            strategy=strategy,
            params=params,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=str(parsed.get("reasoning", "")),
            risk_note=str(parsed.get("risk_note", "")),
            raw=raw[:500],
        )


def make_advisor(name: str = "local", **kwargs) -> Advisor:
    """Build an advisor by name. Unknown names raise rather than degrade."""
    if name == "local":
        return LocalAdvisor()
    if name == "hermes":
        return HermesAdvisor(**kwargs)
    raise KeyError(f"unknown advisor {name!r}; available: ['local', 'hermes']")
