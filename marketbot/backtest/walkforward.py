"""Walk-forward optimization and out-of-sample evaluation.

The point of this module is to make overfitting hard, not to find the best
parameters. Every parameter is selected on a training window and scored only on
the following test window, which the training process never sees. Reported
results are therefore out-of-sample by construction.

It also reports an in-sample versus out-of-sample comparison. When in-sample
results are dramatically better, the strategy is fitting noise, and the report
says so rather than presenting the in-sample number as performance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..config import Config
from ..model import Candle
from ..strategies.base import Strategy
from .engine import BacktestEngine


@dataclass
class WindowResult:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    params: dict
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """True when out-of-sample return is meaningfully worse than in-sample."""
        tr = self.train_metrics.get("total_return")
        te = self.test_metrics.get("total_return")
        if tr is None or te is None:
            return False
        return tr > 0 and te < tr * 0.5


@dataclass
class WalkForwardResult:
    symbol: str
    strategy: str
    windows: list = field(default_factory=list)

    @property
    def oos_return(self) -> Optional[float]:
        vals = [w.test_metrics.get("total_return") for w in self.windows]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def oos_sharpe(self) -> Optional[float]:
        vals = [w.test_metrics.get("sharpe") for w in self.windows]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def oos_win_rate(self) -> Optional[float]:
        vals = []
        for w in self.windows:
            ts = w.test_metrics.get("trade_stats") or {}
            if ts.get("trades"):
                vals.append(ts.get("win_rate"))
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def consistency(self) -> float:
        """Fraction of test windows that were profitable."""
        vals = [w.test_metrics.get("total_return") for w in self.windows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return 0.0
        return sum(1 for v in vals if v > 0) / len(vals)

    @property
    def total_trades(self) -> int:
        return sum((w.test_metrics.get("trade_stats") or {}).get("trades", 0) for w in self.windows)

    def verdict(self) -> str:
        """Plain-language read on whether the strategy holds up out of sample."""
        if len(self.windows) < 2:
            return "insufficient windows to judge"
        degraded = sum(1 for w in self.windows if w.degraded)
        if self.consistency >= 0.6 and degraded <= len(self.windows) / 3:
            return "holds up out of sample"
        if self.consistency < 0.4:
            return "does not hold up: majority of test windows unprofitable"
        return f"mixed: {degraded}/{len(self.windows)} windows degraded sharply"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "windows": [
                {
                    "train": [w.train_start, w.train_end],
                    "test": [w.test_start, w.test_end],
                    "params": w.params,
                    "train": w.train_metrics,
                    "test": w.test_metrics,
                    "degraded": w.degraded,
                }
                for w in self.windows
            ],
            "oos_return": round(self.oos_return, 4) if self.oos_return is not None else None,
            "oos_sharpe": round(self.oos_sharpe, 4) if self.oos_sharpe is not None else None,
            "oos_win_rate": round(self.oos_win_rate, 4) if self.oos_win_rate is not None else None,
            "consistency": round(self.consistency, 4),
            "total_trades": self.total_trades,
            "verdict": self.verdict(),
        }


def grid_search(
    candles: Sequence[Candle],
    strategy_name: str,
    param_grid: dict[str, Sequence],
    config: Optional[Config] = None,
    *,
    score: str = "sharpe",
    min_trades: int = 3,
) -> dict:
    """Exhaustive parameter search over a grid, scored on the whole window.

    Use this only to narrow a search space. It scores in-sample, so its winner
    must be validated with `walk_forward` before being trusted.
    """
    from ..strategies import get

    cfg = config or Config()
    keys = list(param_grid.keys())
    combos = _combos(keys, param_grid)
    scored = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            strategy = get(strategy_name, **params)
            result = BacktestEngine(cfg).run("OPT", candles, strategy)
        except Exception:
            continue
        m = result.metrics
        ts = m.trade_stats
        if ts.get("trades", 0) < min_trades:
            continue
        value = getattr(m, score, None)
        if value is None:
            continue
        scored.append((value, params, m))

    if not scored:
        return {"best_params": None, "best_score": None, "evaluated": len(combos),
                "note": "no parameter set produced enough trades"}
    scored.sort(key=lambda t: t[0], reverse=True)
    best_value, best_params, best_metrics = scored[0]
    return {
        "best_params": best_params,
        "best_score": round(best_value, 4),
        "evaluated": len(scored),
        "total_combos": len(combos),
        "top5": [{"params": p, "score": round(v, 4)} for v, p, _ in scored[:5]],
        "best_metrics": best_metrics.to_dict(),
    }


def _combos(keys: Sequence[str], grid: dict) -> list[list]:
    if not keys:
        return [[]]
    head, *tail = keys
    rest = _combos(tail, grid)
    out = []
    for value in grid[head]:
        for r in rest:
            out.append([value] + r)
    return out


def walk_forward(
    symbol: str,
    candles: Sequence[Candle],
    strategy_name: str,
    param_grid: dict[str, Sequence],
    config: Optional[Config] = None,
    *,
    train_bars: int = 250,
    test_bars: int = 60,
    step: Optional[int] = None,
    score: str = "sharpe",
    min_trades: int = 3,
) -> WalkForwardResult:
    """Optimize on each training window, evaluate on the next test window.

    Windows advance by `step` (default: test_bars), so a long history produces
    many non-overlapping test periods. Parameters are re-fit per window, which
    is what makes the aggregate out-of-sample: no single parameter set is ever
    scored on data it was fitted to.
    """
    cfg = config or Config()
    n = len(candles)
    step = step or test_bars
    result = WalkForwardResult(symbol=symbol, strategy=strategy_name)

    start = train_bars
    while start + test_bars <= n:
        train = candles[start - train_bars : start]
        test = candles[start : start + test_bars]
        if len(train) < train_bars or len(test) < test_bars:
            break

        search = grid_search(train, strategy_name, param_grid, cfg, score=score,
                             min_trades=min_trades)
        params = search.get("best_params") or {}
        from ..strategies import get

        try:
            train_strategy = get(strategy_name, **params)
            train_result = BacktestEngine(cfg).run(symbol, train, train_strategy)
            test_strategy = get(strategy_name, **params)
            test_result = BacktestEngine(cfg).run(symbol, test, test_strategy)
        except Exception:
            start += step
            continue

        result.windows.append(
            WindowResult(
                train_start=start - train_bars,
                train_end=start - 1,
                test_start=start,
                test_end=start + test_bars - 1,
                params=params,
                train_metrics=train_result.metrics.to_dict(),
                test_metrics=test_result.metrics.to_dict(),
            )
        )
        start += step

    return result


def parameter_stability(wf: WalkForwardResult) -> dict:
    """How much the chosen parameters vary across windows.

    A strategy whose 'optimal' parameters swing wildly between windows has no
    stable edge; that is an overfitting signature even when returns look fine.
    """
    if not wf.windows:
        return {"stable": False, "reason": "no windows"}
    param_values: dict[str, list] = {}
    for w in wf.windows:
        for k, v in w.params.items():
            param_values.setdefault(k, []).append(v)
    out = {}
    stable = True
    for key, vals in param_values.items():
        if len(vals) < 2:
            continue
        try:
            sd = statistics.pstdev(vals)
            mean = abs(statistics.mean(vals))
            cv = sd / mean if mean else float("inf")
            out[key] = {"values": vals, "stdev": round(sd, 3), "cv": round(cv, 3)}
            if cv > 0.5:
                stable = False
        except statistics.StatisticsError:
            continue
    return {"stable": stable, "params": out}


def compare_strategies(
    symbol: str,
    candles: Sequence[Candle],
    strategy_names: Sequence[str],
    config: Optional[Config] = None,
) -> list[dict]:
    """Run several strategies over the same window and rank them."""
    from ..strategies import get

    cfg = config or Config()
    out = []
    for name in strategy_names:
        try:
            result = BacktestEngine(cfg).run(symbol, candles, get(name))
        except Exception as e:
            out.append({"strategy": name, "error": str(e)})
            continue
        m = result.metrics
        out.append({
            "strategy": name,
            "total_return": round(m.total_return, 4),
            "sharpe": round(m.sharpe, 3) if m.sharpe is not None else None,
            "max_drawdown": round(m.max_drawdown, 4),
            "trades": m.trade_stats.get("trades", 0),
            "win_rate": m.trade_stats.get("win_rate"),
            "profit_factor": m.trade_stats.get("profit_factor"),
            "directional_accuracy": m.accuracy.get("directional_accuracy"),
            "halted": result.halted,
        })
    out.sort(key=lambda d: (d.get("total_return") or -9), reverse=True)
    return out
