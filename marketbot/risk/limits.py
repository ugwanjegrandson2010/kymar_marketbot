"""Hard trading limits: drawdown kill switch and daily loss limit.

These are enforced by the backtest engine and the live/paper broker before any
order is placed. They exist because a strategy that is quietly bleeding equity
will keep trading through a regime change unless something external stops it.
"""

from __future__ import annotations

from ..config import Config


class RiskLimits:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.halted = False
        self.halt_reason = ""
        self.peak_equity = self.config.starting_cash
        self.session_start_equity = self.config.starting_cash
        self.session_day: int | None = None

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        """Clear a halt. Only call this after the condition genuinely recovered."""
        self.halted = False
        self.halt_reason = ""
        self.session_start_equity = self.peak_equity
        self.session_day = None

    def note_equity(self, equity: float, day: int | None = None) -> None:
        """Track the running peak and roll the daily session boundary."""
        if equity > self.peak_equity:
            self.peak_equity = equity
        if day is not None:
            if self.session_day is None or day != self.session_day:
                self.session_day = day
                self.session_start_equity = equity

    # -- checks ----------------------------------------------------------
    def drawdown_pct(self, equity: float) -> float:
        """Current drawdown from peak, as a positive fraction."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def drawdown_breached(self, equity: float) -> bool:
        return self.drawdown_pct(equity) >= self.config.max_drawdown_pct

    def daily_loss_pct(self, equity: float, session_start_equity: float) -> float:
        if session_start_equity <= 0:
            return 0.0
        return max(0.0, (session_start_equity - equity) / session_start_equity)

    def daily_loss_breached(self, equity: float, session_start_equity: float) -> bool:
        return self.daily_loss_pct(equity, session_start_equity) >= self.config.max_daily_loss_pct

    def check(self, equity: float, day: int | None = None) -> dict:
        """Evaluate every limit and return a verdict. Idempotent.

        Once halted, the bot stays halted until `reset()` is called explicitly;
        re-checking does not silently clear it.

        The daily-loss baseline is captured *before* `note_equity` rolls the
        session forward. Otherwise the first call of a new session compares
        equity against itself and the daily limit can never fire.
        """
        previous_session_start = self.session_start_equity
        self.note_equity(equity, day)
        if self.halted:
            return {"halted": True, "reason": self.halt_reason}

        if self.drawdown_breached(equity):
            self.halted = True
            self.halt_reason = (
                f"max drawdown breached: {self.drawdown_pct(equity):.1%} "
                f">= {self.config.max_drawdown_pct:.1%}"
            )
            return {"halted": True, "reason": self.halt_reason}

        if self.daily_loss_breached(equity, previous_session_start):
            self.halted = True
            self.halt_reason = (
                f"daily loss limit breached: "
                f"{self.daily_loss_pct(equity, previous_session_start):.1%} "
                f">= {self.config.max_daily_loss_pct:.1%}"
            )
            return {"halted": True, "reason": self.halt_reason}

        return {"halted": False, "reason": ""}

    def can_open_position(self, open_count: int) -> bool:
        return open_count < self.config.max_open_positions
