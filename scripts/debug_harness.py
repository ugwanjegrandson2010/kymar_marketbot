"""Systematic error hunt across every MarketBot code path.

Two perspectives, as requested:

  backend  — data, indicators, strategies, backtest engine, risk, metrics,
             walk-forward, cache. Everything that runs without a network.
  liveware — the live trading path: the locked broker, credential handling,
             confirmation gates, and the failure modes when they are not met.

Each check catches its own exceptions and records them, so one broken path does
not hide the others. The exit code is non-zero if any check failed.

Run:  python3 scripts/debug_harness.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []  # (perspective, name, status)


def check(perspective: str, name: str):
    """Decorator that runs a function and records pass/fail/exception."""

    def deco(fn):
        try:
            fn()
            results.append((perspective, name, "PASS"))
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=6)
            results.append((perspective, name, f"FAIL: {type(e).__name__}: {e}\n{tb}"))
        return fn

    return deco


def report() -> int:
    width = max(len(n) for _, n, _ in results)
    fails = 0
    skips = 0
    last_perspective = None
    print()
    for perspective, name, status in results:
        if perspective != last_perspective:
            print(f"\n=== {perspective} ===")
            last_perspective = perspective
        if status.startswith("SKIP"):
            skips += 1
            print(f"  [skip] {name:<{width}}  {status.split(':', 1)[-1].strip()}")
            continue
        mark = "ok  " if status == "PASS" else "FAIL"
        print(f"  [{mark}] {name:<{width}}  {'' if status == 'PASS' else status.splitlines()[0]}")
        if status != "PASS":
            fails += 1
            for line in status.splitlines()[1:]:
                print(f"           {line}")
    print(f"\n{'=' * 70}")
    print(f"{len(results) - fails - skips}/{len(results)} passed, {skips} skipped")
    if fails:
        print(f"{fails} FAILED")
    else:
        print("NO ERRORS FOUND")
    return 1 if fails else 0


# ==========================================================================
# backend
# ==========================================================================
def backend_checks():
    from marketbot.config import load_config
    from marketbot.model import Candle, Signal, SignalAction

    def candles(n=400, drift=0.4, vol=0.3, seed=1):
        import random

        rng = random.Random(seed)
        out = []
        p = 100.0
        for i in range(n):
            o = p
            c = p + drift + rng.uniform(-vol, vol)
            out.append(Candle(ts=1_600_000_000 + i * 86400, open=o,
                              high=max(o, c) + vol, low=min(o, c) - vol,
                              close=c, volume=1000.0 + i))
            p = c
        return out

    @check("backend", "config loads and env overrides apply")
    def _():
        os.environ["MARKETBOT_STARTING_CASH"] = "54321"
        try:
            cfg = load_config()
            assert cfg.starting_cash == 54321.0, cfg.starting_cash
        finally:
            del os.environ["MARKETBOT_STARTING_CASH"]

    @check("backend", "config rejects unknown override")
    def _():
        try:
            load_config(nope=1)
        except AttributeError:
            return
        raise AssertionError("unknown override silently accepted")

    @check("backend", "every indicator computes on a real-shaped series")
    def _():
        from marketbot.indicators import ALL_INDICATORS

        cs = candles(300)
        for name, fn in ALL_INDICATORS.items():
            series = fn(cs, 14)
            assert isinstance(series, list), name
            assert len(series) == len(cs), f"{name}: {len(series)} != {len(cs)}"

    @check("backend", "every indicator is None-safe on tiny input")
    def _():
        from marketbot.indicators import ALL_INDICATORS

        for name, fn in ALL_INDICATORS.items():
            series = fn([Candle(1, 1, 1, 1, 1)], 14)
            assert len(series) == 1, name

    @check("backend", "every indicator survives all-equal prices")
    def _():
        from marketbot.indicators import ALL_INDICATORS

        flat = [Candle(i, 50.0, 50.0, 50.0, 50.0, 0.0) for i in range(100)]
        for name, fn in ALL_INDICATORS.items():
            series = fn(flat, 14)
            for v in series:
                if v is not None:
                    assert v == v, f"{name} produced NaN"  # NaN != NaN

    @check("backend", "every indicator survives zero volume")
    def _():
        from marketbot.indicators import ALL_INDICATORS

        cs = [Candle(i, 10 + i, 11 + i, 9 + i, 10 + i, 0.0) for i in range(100)]
        for name, fn in ALL_INDICATORS.items():
            fn(cs, 14)

    @check("backend", "every strategy instantiates and runs")
    def _():
        from marketbot.backtest import run_backtest
        from marketbot.strategies import available, get

        cs = candles(400)
        for name in available():
            result = run_backtest("T", cs, get(name), load_config())
            assert len(result.equity_curve) == len(cs), name

    @check("backend", "every strategy accepts its own default params explicitly")
    def _():
        from marketbot.strategies import STRATEGIES, available, get

        for name in available():
            cls = STRATEGIES[name]
            get(name, **cls.default_params())

    @check("backend", "every strategy rejects unknown params")
    def _():
        from marketbot.strategies import available, get

        for name in available():
            try:
                get(name, bogus_param=1)
            except ValueError:
                continue
            raise AssertionError(f"{name} accepted a bogus param")

    @check("backend", "engine refuses fewer than 2 candles")
    def _():
        from marketbot.backtest import BacktestEngine
        from marketbot.strategies import get

        try:
            BacktestEngine().run("T", candles(1), get("sma_crossover"))
        except ValueError:
            return
        raise AssertionError("accepted a 1-bar series")

    @check("backend", "engine handles empty candle list")
    def _():
        from marketbot.backtest import BacktestEngine
        from marketbot.strategies import get

        try:
            BacktestEngine().run("T", [], get("sma_crossover"))
        except ValueError:
            return
        raise AssertionError("accepted an empty series")

    @check("backend", "engine handles a series with a zero close")
    def _():
        from marketbot.backtest import BacktestEngine
        from marketbot.strategies import get

        cs = [Candle(i, 10, 11, 9, 10, 100) for i in range(60)]
        cs[30] = Candle(30, 0, 0, 0, 0, 0)
        result = BacktestEngine().run("T", cs, get("sma_crossover"))
        assert result.metrics is not None

    @check("backend", "risk limits halt on drawdown")
    def _():
        from marketbot.backtest import run_backtest
        from marketbot.config import load_config
        from marketbot.strategies.base import Strategy
        from marketbot.model import SignalAction

        class Buys(Strategy):
            name = "buys"
            indicators = (("x", 5),)

            def on_bar(self, i, candles):
                return self.signal(candles, SignalAction.BUY, 1.0, "t")

        cfg = load_config(starting_cash=10_000.0, max_drawdown_pct=0.02)
        falling = candles(400, drift=-5.0, vol=1.0)
        result = run_backtest("T", falling, Buys(), cfg)
        assert result.halted, "drawdown limit never fired"

    @check("backend", "risk limits halt on daily loss")
    def _():
        from marketbot.risk.limits import RiskLimits
        from marketbot.config import load_config

        limits = RiskLimits(load_config(starting_cash=10_000.0, max_daily_loss_pct=0.03))
        verdict = limits.check(9_500.0, day=1)
        assert verdict["halted"], verdict

    @check("backend", "risk limits are idempotent once halted")
    def _():
        from marketbot.risk.limits import RiskLimits
        from marketbot.config import load_config

        limits = RiskLimits(load_config(max_drawdown_pct=0.20))
        limits.check(7_000.0, day=1)
        assert limits.halted
        again = limits.check(9_000.0, day=2)
        assert again["halted"], "halt silently cleared"

    @check("backend", "position sizing never returns negative or inf")
    def _():
        from marketbot.risk.sizing import position_for_risk, size_trade
        from marketbot.config import load_config

        cases = [
            (10_000, 0.01, 100, 90), (10_000, 0.01, 100, 110),
            (0, 0.01, 100, 90), (10_000, 0, 100, 90), (10_000, 0.01, 100, 100),
            (-1, 0.01, 100, 90), (10_000, -1, 100, 90), (1e12, 0.9, 1e-8, 1e-8),
        ]
        for equity, risk, entry, stop in cases:
            qty = position_for_risk(equity, risk, entry, stop)
            assert qty >= 0, (equity, risk, entry, stop, qty)
            assert qty == qty and qty != float("inf"), (equity, risk, entry, stop, qty)
        for entry in (0.0, -1.0, 100.0):
            d = size_trade(10_000, entry, 2.0, load_config())
            assert d.quantity >= 0

    @check("backend", "kelly fraction stays in [0,1] on extreme inputs")
    def _():
        from marketbot.risk.sizing import kelly_fraction

        cases = [(0.0, 1, 1), (1.0, 1, 1), (0.5, 0, 1), (0.5, 1, 0),
                 (0.99, 100, 1), (0.01, 1, 100), (0.5, -1, 1), (0.5, 1, -1)]
        for p, w, l in cases:
            f = kelly_fraction(p, w, l)
            assert 0.0 <= f <= 1.0, (p, w, l, f)

    @check("backend", "metrics handle a flat equity curve")
    def _():
        from marketbot.backtest.metrics import Metrics, sharpe_ratio, max_drawdown

        m = Metrics.from_result("T", [1000.0] * 100, [])
        assert m.sharpe is None, m.sharpe
        assert m.total_return == 0.0
        assert max_drawdown([1000.0] * 100) == 0.0
        assert sharpe_ratio([1000.0]) is None

    @check("backend", "metrics handle an empty trade list")
    def _():
        from marketbot.backtest.metrics import Metrics, trade_stats, accuracy_metrics

        m = Metrics.from_result("T", [1000.0, 1100.0], [])
        assert m.trade_stats["trades"] == 0
        assert trade_stats([])["trades"] == 0
        assert accuracy_metrics([])["directional_accuracy"] is None

    @check("backend", "metrics handle a monotone crash")
    def _():
        from marketbot.backtest.metrics import max_drawdown, sharpe_ratio

        curve = [1000.0 - i for i in range(100)]
        # Falls 1000 -> 901: a 9.9% drawdown, not a total loss.
        dd = max_drawdown(curve)
        assert 0.09 < dd < 0.11, dd
        assert sharpe_ratio(curve) is not None

    @check("backend", "cache round-trips and rejects corrupt rows")
    def _():
        from marketbot.data.cache import CandleCache
        from marketbot.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            cache = CandleCache(load_config(cache_db=Path(tmp) / "c.db"))
            good = [Candle(i, 1, 2, 0, 1, 5) for i in range(20)]
            assert cache.put("p", "S", "1d", good) == 20
            assert len(cache.get("p", "S", "1d")) == 20
            bad = [Candle(1, 0, 0, 0, 0, 0), Candle(2, 1, 2, 0, 1, 5)]
            cache.put("p", "B", "1d", bad)
            assert len(cache.get("p", "B", "1d")) == 1
            cache.close()

    @check("backend", "cache survives a zero-length and empty put")
    def _():
        from marketbot.data.cache import CandleCache
        from marketbot.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            cache = CandleCache(load_config(cache_db=Path(tmp) / "c.db"))
            assert cache.put("p", "S", "1d", []) == 0
            assert cache.get("p", "S", "1d") == []
            assert cache.coverage("p", "S", "1d") is None
            cache.close()

    @check("backend", "walk-forward produces out-of-sample windows")
    def _():
        from marketbot.backtest.walkforward import walk_forward, parameter_stability
        from marketbot.strategies import get
        from marketbot.config import load_config

        cs = candles(1200)
        wf = walk_forward("T", cs, "sma_crossover", {"fast": [5, 10], "slow": [30, 50]},
                          load_config(), train_bars=250, test_bars=60)
        assert len(wf.windows) >= 2, len(wf.windows)
        for w in wf.windows:
            assert w.test_end < len(cs)
        stability = parameter_stability(wf)
        assert "stable" in stability

    @check("backend", "walk-forward handles a series too short to split")
    def _():
        from marketbot.backtest.walkforward import walk_forward
        from marketbot.config import load_config

        wf = walk_forward("T", candles(50), "sma_crossover", {"fast": [5, 10]},
                          load_config(), train_bars=250, test_bars=60)
        assert wf.windows == []

    @check("backend", "grid search handles an empty grid")
    def _():
        from marketbot.backtest.walkforward import grid_search
        from marketbot.config import load_config

        result = grid_search(candles(200), "sma_crossover", {}, load_config())
        assert "best_params" in result

    @check("backend", "compare_strategies reports every strategy")
    def _():
        from marketbot.backtest.walkforward import compare_strategies
        from marketbot.config import load_config
        from marketbot.strategies import available

        rows = compare_strategies("T", candles(400), available(), load_config())
        assert len(rows) == len(available()), (len(rows), len(available()))

    @check("backend", "universe builds, saves, and loads")
    def _():
        from marketbot.data import Universe
        from marketbot.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            u = Universe(load_config(cache_db=Path(tmp) / "c.db"))
            assets = u.defaults(include_stocks=True)
            assert assets
            path = u.save(assets, Path(tmp) / "u.json")
            loaded = u.load(path)
            assert [a.symbol for a in loaded] == [a.symbol for a in assets]
            u.close()

    @check("backend", "every strategy's signal has a valid action and bounds")
    def _():
        from marketbot.model import SignalAction
        from marketbot.strategies import available, get

        cs = candles(400)
        for name in available():
            s = get(name)
            s.prepare(cs)
            for i in range(5, len(cs), 25):
                sig = s.on_bar(i, cs[: i + 1])
                assert isinstance(sig.action, SignalAction), name
                assert 0.0 <= sig.strength <= 1.0, (name, sig.strength)

    @check("backend", "composite strategy vote is bounded and explainable")
    def _():
        from marketbot.strategies import get

        cs = candles(400)
        s = get("composite")
        s.prepare(cs)
        score, votes = s.score_at(300)
        assert -1.0 <= score <= 1.0, score
        assert votes > 0
        breakdown = s.vote_breakdown(300)
        assert breakdown, "vote breakdown empty"

    @check("backend", "composite tolerates an empty weight set")
    def _():
        """An empty weight map means uniform weighting, not a crash."""
        from marketbot.strategies import get

        cs = candles(400)
        s = get("composite", weights={})
        s.prepare(cs)
        score, votes = s.score_at(300)
        assert votes > 0, "no indicators voted"
        assert -1.0 <= score <= 1.0, score
        # Uniform weighting must equal the mean of the *signed* normalized
        # readings over the indicators that actually voted.
        from marketbot.strategies.meta import normalize

        norms = [normalize(n, s._series[n][300], s._cl[300]) for n in s._names]
        norms = [x for x in norms if x is not None]
        assert norms, "no normalizable indicators"
        expected = sum(norms) / len(norms)
        assert abs(score - expected) < 1e-6, (score, expected)

    @check("backend", "firm runs end to end")
    def _():
        from marketbot.ai import AnalysisRequest, make_firm
        from marketbot.config import load_config

        req = AnalysisRequest(symbol="T", candles=candles(400),
                              strategy_options=["composite"])
        result = make_firm(load_config()).run(req)
        assert len(result["analysts"]) == 5
        assert "risk_approved" in result["decision"]
        assert isinstance(result["decision"]["executable"], bool)

    @check("backend", "risk veto survives portfolio approval")
    def _():
        from marketbot.ai.multiagent import TradeDecision, RiskManager, PortfolioManager
        from marketbot.ai import AnalysisRequest
        from marketbot.config import load_config
        from marketbot.model import SignalAction

        req = AnalysisRequest(symbol="T", candles=candles(400), strategy_options=["x"],
                              constraints={"open_positions": 99})
        decision = TradeDecision(SignalAction.BUY, "composite", {}, 0.9, "test")
        reviewed = RiskManager(load_config(max_open_positions=3)).review(decision, req)
        assert not reviewed.risk_approved
        approved = PortfolioManager().approve(reviewed, req)
        assert not approved.executable

    @check("backend", "hermes advisor rejects malformed replies")
    def _():
        from marketbot.ai.advisor import HermesAdvisor
        from marketbot.ai import AnalysisRequest
        from marketbot.model import SignalAction

        req = AnalysisRequest(symbol="T", candles=candles(400), strategy_options=["composite"])
        for payload in ({"action": "nonsense"}, {"action": "buy", "strategy": "bogus"},
                        {"action": "buy", "confidence": "not-a-number"}):
            advice = HermesAdvisor()._validate(payload, req, "{}")
            assert advice.action is SignalAction.HOLD, payload

    @check("backend", "broker accounting stays self-consistent")
    def _():
        from marketbot.broker.paper import PaperBroker
        from marketbot.config import load_config
        from marketbot.model import Candle, Order, OrderType, Side

        cfg = load_config(starting_cash=10_000.0, fee_rate=0.001, slippage_rate=0.0)
        broker = PaperBroker(cfg.starting_cash, cfg.fee_rate, cfg.slippage_rate)
        order = Order(symbol="T", side=Side.BUY, quantity=10, type=OrderType.MARKET, ts=1)
        broker.submit(order)
        broker.tick(Candle(2, 100, 101, 99, 100, 1000, symbol="T"), 1)
        assert broker.cash < 10_000.0
        assert broker.positions["T"].quantity == 10
        equity = broker.equity({"T": 105.0})
        assert equity > 10_000.0, equity  # bought at 100, now worth 1050

    @check("backend", "broker rejects an unaffordable order")
    def _():
        from marketbot.broker.paper import PaperBroker
        from marketbot.model import Candle, Order, OrderType, OrderStatus, Side

        broker = PaperBroker(100.0, 0.001, 0.0)
        broker.submit(Order(symbol="T", side=Side.BUY, quantity=1000, type=OrderType.MARKET))
        broker.tick(Candle(2, 100, 101, 99, 100, 1, symbol="T"), 1)
        assert broker.orders[-1].status is OrderStatus.REJECTED

    @check("backend", "broker rejects shorts when disabled")
    def _():
        from marketbot.broker.paper import PaperBroker
        from marketbot.model import Order, OrderType, OrderStatus, Side

        broker = PaperBroker(10_000.0, 0.001, 0.0, allow_short=False)
        broker.submit(Order(symbol="T", side=Side.SELL, quantity=1, type=OrderType.MARKET))
        assert broker.orders[-1].status is OrderStatus.REJECTED

    @check("backend", "broker handles a limit order that never fills")
    def _():
        from marketbot.broker.paper import PaperBroker
        from marketbot.model import Candle, Order, OrderType, OrderStatus, Side

        broker = PaperBroker(10_000.0, 0.0, 0.0)
        broker.submit(Order(symbol="T", side=Side.BUY, quantity=1, type=OrderType.LIMIT,
                            limit_price=50.0))
        broker.tick(Candle(2, 100, 101, 99, 100, 1, symbol="T"), 1)
        assert broker.orders[-1].status is OrderStatus.PENDING
        broker.tick(Candle(3, 50, 51, 49, 50, 1, symbol="T"), 2)
        assert broker.orders[-1].status is OrderStatus.FILLED

    @check("backend", "broker handles a stop order")
    def _():
        from marketbot.broker.paper import PaperBroker
        from marketbot.model import Candle, Order, OrderType, OrderStatus, Side

        broker = PaperBroker(10_000.0, 0.0, 0.0)
        broker.submit(Order(symbol="T", side=Side.BUY, quantity=1, type=OrderType.STOP,
                            stop_price=110.0))
        broker.tick(Candle(2, 100, 101, 99, 100, 1, symbol="T"), 1)
        assert broker.orders[-1].status is OrderStatus.PENDING
        broker.tick(Candle(3, 111, 112, 110, 111, 1, symbol="T"), 2)
        assert broker.orders[-1].status is OrderStatus.FILLED

    @check("backend", "broker P&L is correct for a long and a short round trip")
    def _():
        from marketbot.broker.paper import PaperBroker
        from marketbot.model import Candle, Order, OrderType, Side

        for side, prices, expected in (
            (Side.BUY, (100.0, 110.0), 10.0),
            (Side.SELL, (100.0, 90.0), 10.0),
        ):
            broker = PaperBroker(10_000.0, 0.0, 0.0, allow_short=True)
            broker.submit(Order(symbol="T", side=side, quantity=1, type=OrderType.MARKET))
            broker.tick(Candle(2, prices[0], prices[0], prices[0], prices[0], 1, symbol="T"), 1)
            broker.submit(Order(symbol="T",
                                side=Side.SELL if side is Side.BUY else Side.BUY,
                                quantity=1, type=OrderType.MARKET))
            broker.tick(Candle(3, prices[1], prices[1], prices[1], prices[1], 1, symbol="T"), 2)
            assert len(broker.trades) == 1, side
            assert abs(broker.trades[0].pnl - expected) < 1e-6, (side, broker.trades[0].pnl)

    @check("backend", "cli parses every documented command")
    def _():
        from marketbot.cli import build_parser

        parser = build_parser()
        for argv in (["backtest", "--symbol", "BTC-USD", "--strategy", "sma_crossover"],
                     ["compare", "--symbol", "ETH-USD"],
                     ["optimize", "--strategy", "sma_crossover", "--symbol", "BTC-USD"],
                     ["paper", "--strategy", "sma_crossover"],
                     ["firm", "--symbol", "BTC-USD"],
                     ["advise", "--symbol", "BTC-USD"],
                     ["indicators", "--list"],
                     ["universe"],
                     ["live-status"]):
            args = parser.parse_args(argv)
            assert hasattr(args, "func"), argv

    @check("backend", "cli rejects an unknown command")
    def _():
        from marketbot.cli import build_parser

        try:
            build_parser().parse_args(["not_a_command"])
        except SystemExit:
            return
        raise AssertionError("unknown command accepted")

    @check("backend", "api module imports and builds the app")
    def _():
        from marketbot.api import create_app

        app = create_app()
        assert app is not None

    @check("backend", "tui module constructs")
    def _():
        from marketbot.tui import TUI, SCREENS

        tui = TUI()
        assert len(SCREENS) == 7
        assert tui.symbol

    @check("backend", "all subpackages import cleanly")
    def _():
        import importlib

        for name in ("config", "model", "data", "data.base", "data.cache",
                     "data.coingecko", "data.yahoo", "data.universe", "indicators",
                     "strategies", "strategies.base", "strategies.trend",
                     "strategies.momentum", "strategies.volatility", "strategies.volume",
                     "strategies.legendary", "strategies.meta", "backtest",
                     "backtest.engine", "backtest.metrics", "backtest.walkforward",
                     "risk", "risk.sizing", "risk.limits", "broker", "broker.base",
                     "broker.paper", "broker.live", "ai", "ai.advisor", "ai.multiagent",
                     "cli", "tui"):
            importlib.import_module(f"marketbot.{name}")


# ==========================================================================
# liveware
# ==========================================================================
def liveware_checks():
    @check("liveware", "live trading is refused by default")
    def _():
        from marketbot.broker.live import live_ok, load_broker
        from marketbot.broker.paper import PaperBroker

        allowed, reason = live_ok()
        assert not allowed, reason
        broker = load_broker()
        assert isinstance(broker, PaperBroker), type(broker)

    @check("liveware", "live_ok requires the environment flag")
    def _():
        from marketbot.broker.live import live_ok

        old = os.environ.pop("MARKETBOT_LIVE_TRADING", None)
        try:
            allowed, reason = live_ok(confirmation="I understand this trades real money")
            assert not allowed
            assert "LIVE_TRADING" in reason
        finally:
            if old is not None:
                os.environ["MARKETBOT_LIVE_TRADING"] = old

    @check("liveware", "flag alone is not enough without credentials")
    def _():
        from marketbot.broker.live import live_ok

        old_flag = os.environ.get("MARKETBOT_LIVE_TRADING")
        old_key = os.environ.pop("MARKETBOT_API_KEY", None)
        os.environ["MARKETBOT_LIVE_TRADING"] = "1"
        try:
            allowed, reason = live_ok(confirmation="I understand this trades real money")
            assert not allowed
            assert "credential" in reason or "exchange" in reason, reason
        finally:
            if old_flag is not None:
                os.environ["MARKETBOT_LIVE_TRADING"] = old_flag
            else:
                os.environ.pop("MARKETBOT_LIVE_TRADING", None)
            if old_key is not None:
                os.environ["MARKETBOT_API_KEY"] = old_key

    @check("liveware", "credentials alone are not enough without the flag")
    def _():
        from marketbot.broker.live import live_ok

        old_flag = os.environ.pop("MARKETBOT_LIVE_TRADING", None)
        os.environ["MARKETBOT_EXCHANGE"] = "binance"
        os.environ["MARKETBOT_API_KEY"] = "x"
        os.environ["MARKETBOT_SECRET"] = "y"
        try:
            allowed, _ = live_ok(confirmation="I understand this trades real money")
            assert not allowed
        finally:
            os.environ.pop("MARKETBOT_EXCHANGE", None)
            os.environ.pop("MARKETBOT_API_KEY", None)
            os.environ.pop("MARKETBOT_SECRET", None)
            if old_flag is not None:
                os.environ["MARKETBOT_LIVE_TRADING"] = old_flag

    @check("liveware", "the wrong confirmation phrase is refused")
    def _():
        from marketbot.broker.live import live_ok

        old_flag = os.environ.get("MARKETBOT_LIVE_TRADING")
        old_key = os.environ.get("MARKETBOT_API_KEY")
        old_sec = os.environ.get("MARKETBOT_SECRET")
        os.environ["MARKETBOT_LIVE_TRADING"] = "1"
        os.environ["MARKETBOT_EXCHANGE"] = "binance"
        os.environ["MARKETBOT_API_KEY"] = "x"
        os.environ["MARKETBOT_SECRET"] = "y"
        try:
            for phrase in ("", "yes", "I agree", "i understand this trades real money"):
                allowed, reason = live_ok(confirmation=phrase)
                assert not allowed, phrase
                assert "confirmation" in reason, reason
            allowed, _ = live_ok(confirmation="I understand this trades real money")
            assert allowed, "correct phrase refused"
        finally:
            for key, val in (("MARKETBOT_LIVE_TRADING", old_flag),
                             ("MARKETBOT_API_KEY", old_key),
                             ("MARKETBOT_SECRET", old_sec)):
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)
            os.environ.pop("MARKETBOT_EXCHANGE", None)

    @check("liveware", "LiveBroker construction is refused without the phrase")
    def _():
        from marketbot.broker.live import LiveBroker
        from marketbot.broker.base import BrokerError

        try:
            LiveBroker()
        except BrokerError:
            return
        raise AssertionError("LiveBroker constructed without gates")

    @check("liveware", "LiveBroker construction is refused with the wrong phrase")
    def _():
        from marketbot.broker.live import LiveBroker
        from marketbot.broker.base import BrokerError

        try:
            LiveBroker(confirmation="let me in")
        except BrokerError:
            return
        raise AssertionError("LiveBroker accepted a wrong confirmation phrase")

    @check("liveware", "live status never leaks a secret")
    def _():
        import json

        from marketbot.broker.live import describe_live_status, credentials

        os.environ["MARKETBOT_API_KEY"] = "SUPER-SECRET-KEY-123"
        try:
            status = describe_live_status()
            blob = json.dumps(status)
            assert "SUPER-SECRET-KEY" not in blob, blob
            assert status["has_key"] is True
            creds = credentials()
            assert "SUPER-SECRET-KEY" not in json.dumps(
                {k: v for k, v in creds.items() if k in ("exchange", "sandbox")}
            )
        finally:
            os.environ.pop("MARKETBOT_API_KEY", None)

    @check("liveware", "load_broker never returns a live broker by accident")
    def _():
        from marketbot.broker.live import load_broker
        from marketbot.broker.paper import PaperBroker

        old_flag = os.environ.pop("MARKETBOT_LIVE_TRADING", None)
        try:
            for _ in range(3):
                broker = load_broker()
                assert isinstance(broker, PaperBroker)
        finally:
            if old_flag is not None:
                os.environ["MARKETBOT_LIVE_TRADING"] = old_flag

    @check("liveware", "describe_live_status works with no configuration at all")
    def _():
        from marketbot.broker.live import describe_live_status

        status = describe_live_status()
        assert status["live_enabled"] is False
        assert status["reason"]

    @check("liveware", "config defaults to paper")
    def _():
        from marketbot.config import load_config

        old = os.environ.pop("MARKETBOT_LIVE_TRADING", None)
        try:
            assert load_config().live_trading is False
            os.environ["MARKETBOT_LIVE_TRADING"] = "0"
            assert load_config().live_trading is False
        finally:
            if old is not None:
                os.environ["MARKETBOT_LIVE_TRADING"] = old
            else:
                os.environ.pop("MARKETBOT_LIVE_TRADING", None)

    @check("liveware", "an exchange error during submit is rejected, not raised")
    def _():
        """A failing live order must surface as a rejected order, not a crash.

        Skipped when ccxt is absent: ccxt is a genuine hard requirement for live
        trading, and this check exercises the exchange-error path specifically.
        """
        try:
            import ccxt  # noqa: F401
        except ImportError:
            results.append(("liveware", "an exchange error during submit is rejected, not raised",
                            "SKIP: ccxt not installed (required for live trading)"))
            return

        from marketbot.broker.live import LiveBroker
        from marketbot.broker.base import Broker
        from marketbot.model import Order, OrderType, OrderStatus, Side

        class FakeExchange:
            def create_order(self, *a, **kw):
                raise RuntimeError("network down")

            def set_sandbox_mode(self, *a):
                pass

        broker = LiveBroker.__new__(LiveBroker)
        Broker.__init__(broker, cash=0.0, fee_rate=0.001, slippage_rate=0.0, allow_short=True)
        broker.exchange = FakeExchange()
        broker.sandbox = False
        order = Order(symbol="BTC/USD", side=Side.BUY, quantity=1, type=OrderType.MARKET)
        result = broker.submit(order)
        assert result.status is OrderStatus.REJECTED, result.status
        assert "network down" in result.reject_reason

    @check("liveware", "market id conversion is correct")
    def _():
        from marketbot.broker.live import LiveBroker

        broker = LiveBroker.__new__(LiveBroker)
        assert broker._market_for("BTC-USD") == "BTC/USD"
        assert broker._market_for("ETH-USD") == "ETH/USD"

    @check("liveware", "live broker tick does not fabricate fills")
    def _():
        from marketbot.broker.live import LiveBroker
        from marketbot.model import Candle

        broker = LiveBroker.__new__(LiveBroker)
        broker._prices = {}
        fills = broker.tick(Candle(1, 100, 101, 99, 100, 1, symbol="BTC/USD"), 0)
        assert fills == []
        assert broker.get_price("BTC/USD") == 100.0


if __name__ == "__main__":
    print("MarketBot error hunt")
    print("=" * 70)
    backend_checks()
    liveware_checks()
    raise SystemExit(report())
