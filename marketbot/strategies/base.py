"""Strategy interface.

A strategy sees the candle history strictly up to and including bar `i`, and
returns one Signal for that bar. It never sees bar i+1. The engine is what
executes that signal at the next bar's open, which is what makes the whole
system's no-look-ahead property hold.

Strategies are pure functions of price history plus their parameters: no
internal state that persists across bars is allowed to carry information from
the future. Anything stateful (e.g. a pending stop) belongs in the broker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from ..model import Candle, Signal, SignalAction


class Strategy(ABC):
    """Base class for all trading strategies."""

    name = "base"
    description = ""
    # Indicators this strategy needs, as (name, period) pairs. The engine can
    # pre-compute them so a strategy does not recompute on every bar.
    indicators: tuple = ()

    def __init__(self, **params: Any):
        self.params = {**self.default_params(), **params}
        self._cache: dict = {}

    @classmethod
    def default_params(cls) -> dict:
        return {}

    # -- lifecycle -------------------------------------------------------
    def prepare(self, candles: Sequence[Candle]) -> None:
        """Optional hook to precompute indicator series over the whole window.

        Called once before the bar loop. Implementations may cache results here;
        the values must be functions of the full history only in the sense that
        `on_bar` still indexes them by i, never reads ahead.
        """
        return None

    @abstractmethod
    def on_bar(self, i: int, candles: Sequence[Candle]) -> Signal:
        """Return the action for bar i given candles[:i+1]."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    def signal(
        self,
        candles: Sequence[Candle],
        action: SignalAction,
        strength: float = 1.0,
        reason: str = "",
        **indicator_values: Optional[float],
    ) -> Signal:
        """Build a Signal stamped with the bar's timestamp and symbol."""
        bar = candles[-1]
        return Signal(
            action=action,
            symbol=bar.symbol,
            ts=bar.ts,
            strength=max(0.0, min(1.0, float(strength))),
            reason=reason,
            indicators={k: v for k, v in indicator_values.items() if v is not None},
        )

    def cached(self, key: str, factory) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def param(self, key: str, default=None):
        return self.params.get(key, default)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} {self.params}>"


def validate_params(strategy_cls, params: dict) -> dict:
    """Merge params over defaults and reject unknown keys.

    A typo in a parameter name would otherwise silently fall back to the default
    and produce a backtest that does not test what the operator asked for.
    """
    merged = {**strategy_cls.default_params(), **params}
    unknown = set(params) - set(strategy_cls.default_params())
    if unknown:
        raise ValueError(
            f"{strategy_cls.__name__} got unknown params {sorted(unknown)}; "
            f"valid: {sorted(strategy_cls.default_params())}"
        )
    return merged
