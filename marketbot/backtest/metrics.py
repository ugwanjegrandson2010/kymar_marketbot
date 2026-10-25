"""Backtest and live-trading performance metrics.

Every metric here is computed from the completed trade list and the equity
curve only. None of them read the price series directly, so a metric cannot be
inadvertently fed future information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..model import Trade


def _returns(equity_curve: Sequence[float]) -> list[float]:
    out = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        out.append((equity_curve[i] - prev) / prev if prev else 0.0)
    return out


def sharpe_ratio(equity_curve: Sequence[float], periods_per_year: int = 365) -> Optional[float]:
    """Annualized Sharpe using the mean excess return over its own stdev.

    Returns None when there are fewer than two equity points or zero variance,
    rather than returning a meaningless 0.0 or dividing by zero.
    """
    rets = _returns(equity_curve)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mean / sd) * math.sqrt(periods_per_year)


def sortino_ratio(equity_curve: Sequence[float], periods_per_year: int = 365) -> Optional[float]:
    """Annualized Sortino: mean return over downside deviation only."""
    rets = _returns(equity_curve)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    downside = [r for r in rets if r < 0]
    if not downside:
        return None
    var = sum(r * r for r in downside) / len(downside)
    dd = math.sqrt(var)
    if dd == 0:
        return None
    return (mean / dd) * math.sqrt(periods_per_year)


def drawdown_series(equity_curve: Sequence[float]) -> list[float]:
    """Running drawdown from the peak, as a positive fraction, per point."""
    out: list[float] = []
    peak = -math.inf
    for v in equity_curve:
        peak = max(peak, v)
        out.append((peak - v) / peak if peak > 0 else 0.0)
    return out


def max_drawdown(equity_curve: Sequence[float]) -> float:
    return max(drawdown_series(equity_curve)) if equity_curve else 0.0


def calmar_ratio(
    equity_curve: Sequence[float], periods_per_year: int = 365
) -> Optional[float]:
    """Annualized return divided by max drawdown; None when there is no drawdown."""
    dd = max_drawdown(equity_curve)
    if dd == 0 or len(equity_curve) < 2:
        return None
    ann = annualized_return(equity_curve, periods_per_year)
    if ann is None:
        return None
    return ann / dd


def annualized_return(
    equity_curve: Sequence[float], periods_per_year: int = 365
) -> Optional[float]:
    """Compound annual growth rate from the first and last equity values."""
    if len(equity_curve) < 2:
        return None
    start, end = equity_curve[0], equity_curve[-1]
    if start <= 0:
        return None
    years = (len(equity_curve) - 1) / periods_per_year
    if years <= 0:
        return None
    return (end / start) ** (1 / years) - 1.0


def total_return(equity_curve: Sequence[float]) -> float:
    if not equity_curve or equity_curve[0] == 0:
        return 0.0
    return equity_curve[-1] / equity_curve[0] - 1.0


def trade_stats(trades: Sequence[Trade]) -> dict:
    """Win rate, profit factor, expectancy, and averages from completed trades."""
    n = len(trades)
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    win_rate = (len(wins) / n) if n else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else None
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(gross_win - gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 4) if payoff is not None else None,
        "expectancy": round(
            win_rate * avg_win - (1 - win_rate) * avg_loss, 2
        ) if n else 0.0,
        "avg_bars_held": round(sum(t.bars_held for t in trades) / n, 1) if n else 0.0,
        "total_fees": round(sum(t.fees for t in trades), 2),
    }


def accuracy_metrics(trades: Sequence[Trade]) -> dict:
    """Directional accuracy: how often the traded direction was right.

    Distinct from win rate, which measures profitability. A strategy can have
    low directional accuracy and still profit through asymmetry (few large wins,
    many small losses), and the two should never be conflated.
    """
    if not trades:
        return {
            "directional_accuracy": None,
            "long_accuracy": None,
            "short_accuracy": None,
            "long_trades": 0,
            "short_trades": 0,
        }
    correct = 0
    long_trades = [t for t in trades if t.side.value == "buy"]
    short_trades = [t for t in trades if t.side.value == "sell"]
    long_correct = sum(1 for t in long_trades if t.pnl > 0)
    short_correct = sum(1 for t in short_trades if t.pnl > 0)
    correct = long_correct + short_correct
    return {
        "directional_accuracy": round(correct / len(trades), 4),
        "long_accuracy": round(long_correct / len(long_trades), 4) if long_trades else None,
        "short_accuracy": round(short_correct / len(short_trades), 4) if short_trades else None,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
    }


@dataclass
class Metrics:
    symbol: str = ""
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    total_return: float = 0.0
    annualized_return: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: float = 0.0
    calmar: Optional[float] = None
    trade_stats: dict = field(default_factory=dict)
    accuracy: dict = field(default_factory=dict)
    periods_per_year: int = 365

    @classmethod
    def from_result(
        cls,
        symbol: str,
        equity_curve: Sequence[float],
        trades: Sequence[Trade],
        periods_per_year: int = 365,
    ) -> "Metrics":
        return cls(
            symbol=symbol,
            equity_curve=list(equity_curve),
            returns=_returns(equity_curve),
            total_return=total_return(equity_curve),
            annualized_return=annualized_return(equity_curve, periods_per_year),
            sharpe=sharpe_ratio(equity_curve, periods_per_year),
            sortino=sortino_ratio(equity_curve, periods_per_year),
            max_drawdown=max_drawdown(equity_curve),
            calmar=calmar_ratio(equity_curve, periods_per_year),
            trade_stats=trade_stats(trades),
            accuracy=accuracy_metrics(trades),
            periods_per_year=periods_per_year,
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_return": round(self.total_return, 4),
            "annualized_return": round(self.annualized_return, 4)
            if self.annualized_return is not None
            else None,
            "sharpe": round(self.sharpe, 4) if self.sharpe is not None else None,
            "sortino": round(self.sortino, 4) if self.sortino is not None else None,
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar": round(self.calmar, 4) if self.calmar is not None else None,
            "trade_stats": self.trade_stats,
            "accuracy": self.accuracy,
        }

    def summary(self) -> str:
        ts = self.trade_stats
        return (
            f"{self.symbol}: return {self.total_return:+.1%} | "
            f"sharpe {self.sharpe if self.sharpe is None else round(self.sharpe, 2)} | "
            f"maxDD {self.max_drawdown:.1%} | "
            f"trades {ts.get('trades', 0)} winrate {ts.get('win_rate', 0):.0%} | "
            f"PF {ts.get('profit_factor')}"
        )
