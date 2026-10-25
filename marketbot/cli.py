"""Command-line interface.

Every command is read-only with respect to real money unless it goes through
`broker/live.py`, which cannot be satisfied accidentally. `paper` runs the
strategy against live data through the simulated broker; `live` refuses unless
every gate is passed.

Run `python -m marketbot --help` for the full command list.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .config import load_config


def _print(data, as_json: bool = True) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)


def cmd_universe(args) -> int:
    from .data import Universe

    with Universe(args.config) as u:
        if args.build:
            assets = u.build_default(
                include_stocks=not args.crypto_only, long_tail_pages=args.long_tail_pages
            )
            print(f"universe: {len(assets)} assets -> {u.config.cache_db.parent / 'universe.json'}")
        assets = u.load() or u.defaults(include_stocks=not args.crypto_only)
        if args.verify:
            report = u.verify(assets, interval=args.interval, min_bars=args.min_bars)
            _print({
                "ok": len(report["ok"]),
                "failed": len(report["failed"]),
                "failures": report["failed"][:20],
            })
        else:
            _print([a.symbol for a in assets])
    return 0


def cmd_indicators(args) -> int:
    from .indicators import available, compute

    if args.list:
        _print({"count": len(available()), "indicators": available()})
        return 0
    if not args.symbol:
        print("error: --symbol is required unless --list is used", file=sys.stderr)
        return 2
    from .data import Universe

    with Universe(args.config) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    for name in (args.name or ["rsi", "sma", "macd", "atr"]):
        series = compute(name, candles, args.period)
        defined = [v for v in series if v is not None]
        last = series[-1]
        print(
            f"{name:14} period={args.period:<4} defined={len(defined):>5} "
            f"last={last if last is None else round(last, 6)}"
        )
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run_backtest
    from .data import Universe
    from .strategies import get

    strategy = get(args.strategy, **(json.loads(args.params) if args.params else {}))
    with Universe(args.config) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    result = run_backtest(args.symbol, candles, strategy, args.config)
    if args.json:
        _print(result.to_dict())
    else:
        from .report import backtest_report

        print(backtest_report(result))
    return 0


def cmd_compare(args) -> int:
    from .backtest.walkforward import compare_strategies
    from .data import Universe
    from .strategies import available

    names = args.strategies.split(",") if args.strategies else available()
    with Universe(args.config) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    rows = compare_strategies(args.symbol, candles, names, args.config)
    if args.json:
        _print(rows)
    else:
        from .report import compare_report

        print(compare_report(rows))
    return 0


def cmd_optimize(args) -> int:
    from .backtest.walkforward import parameter_stability, walk_forward
    from .data import Universe

    grid = json.loads(args.grid) if args.grid else {"period": [10, 20, 50]}
    with Universe(args.config) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    wf = walk_forward(
        args.symbol, candles, args.strategy, grid, args.config,
        train_bars=args.train, test_bars=args.test, score=args.score,
    )
    if args.json:
        _print({**wf.to_dict(), "stability": parameter_stability(wf)})
    else:
        d = wf.to_dict()
        print(f"windows: {len(wf.windows)}  total trades: {d['total_trades']}")
        print(f"out-of-sample return: {d['oos_return']}")
        print(f"out-of-sample sharpe: {d['oos_sharpe']}")
        print(f"consistency: {d['consistency']:.0%}")
        print(f"verdict: {d['verdict']}")
    return 0


def cmd_paper(args) -> int:
    """Run the strategy against live data through the simulated broker."""
    from .broker.paper import PaperBroker
    from .data import Universe
    from .strategies import get

    cfg = args.config
    strategy = get(args.strategy, **(json.loads(args.params) if args.params else {}))
    with Universe(cfg) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    broker = PaperBroker(cfg.starting_cash, cfg.fee_rate, cfg.slippage_rate, cfg.allow_short)
    from .backtest.engine import BacktestEngine

    result = BacktestEngine(cfg, broker).run(args.symbol, candles, strategy)
    _print({
        "mode": "paper",
        "symbol": args.symbol,
        "strategy": args.strategy,
        "final_equity": round(result.equity_curve[-1], 2),
        "return": round(result.metrics.total_return, 4),
        "metrics": result.metrics.to_dict(),
        "positions": broker.to_dict()["positions"],
    })
    return 0


def cmd_live_status(args) -> int:
    from .broker.live import describe_live_status

    _print(describe_live_status())
    return 0


def cmd_advise(args) -> int:
    from .ai import AnalysisRequest, make_advisor
    from .data import Universe
    from .strategies import available

    with Universe(args.config) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    request = AnalysisRequest(
        symbol=args.symbol,
        candles=candles,
        strategy_options=available(),
        question=args.question or "What should I trade?",
    )
    advisor = make_advisor(args.advisor)
    advice = advisor.advise(request)
    _print(advice.to_dict())
    return 0


def cmd_firm(args) -> int:
    """Run the multi-agent trading firm over one symbol."""
    from .ai import AnalysisRequest, make_firm
    from .data import Universe
    from .strategies import available

    with Universe(args.config) as u:
        candles = u.yahoo.fetch_candles(args.symbol, interval=args.interval, limit=args.bars)
    request = AnalysisRequest(
        symbol=args.symbol,
        candles=candles,
        strategy_options=available(),
        constraints={"kind": args.kind},
    )
    firm = make_firm(args.config, advisor_name=args.advisor)
    result = firm.run(request)
    if args.json:
        _print(result)
    else:
        from .report import firm_report

        print(firm_report(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="marketbot",
        description="Backtest, paper trade, and (explicitly) live trade crypto and stocks.",
    )
    p.add_argument("--version", action="version", version=f"marketbot {__version__}")
    p.add_argument("--data-dir", default=None, help="override the data/cache directory")
    p.add_argument("--cache-db", default=None, help="override the SQLite cache path")

    def add_common(sp):
        sp.add_argument("--symbol", default="BTC-USD")
        sp.add_argument("--interval", default="1d")
        sp.add_argument("--bars", type=int, default=400)

    sub = p.add_subparsers(dest="command", required=True)

    u = sub.add_parser("universe", help="build or verify the tradable universe")
    u.add_argument("--build", action="store_true")
    u.add_argument("--verify", action="store_true")
    u.add_argument("--long-tail-pages", type=int, default=0)
    u.add_argument("--crypto-only", action="store_true")
    u.add_argument("--min-bars", type=int, default=60)
    u.set_defaults(func=cmd_universe)

    i = sub.add_parser("indicators", help="list or compute indicators")
    i.add_argument("--list", action="store_true")
    i.add_argument("--name", action="append")
    i.add_argument("--period", type=int, default=14)
    i.set_defaults(func=cmd_indicators)

    b = sub.add_parser("backtest", help="backtest one strategy on one symbol")
    b.add_argument("--strategy", required=True)
    b.add_argument("--params", default=None, help='JSON object, e.g. \'{"fast":10}\'')
    b.add_argument("--json", action="store_true")
    add_common(b)
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("compare", help="rank strategies on one symbol")
    c.add_argument("--strategies", default=None, help="comma-separated subset")
    add_common(c)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)

    o = sub.add_parser("optimize", help="walk-forward optimize a strategy")
    o.add_argument("--strategy", required=True)
    o.add_argument("--grid", default=None, help='JSON object of param lists')
    o.add_argument("--train", type=int, default=250)
    o.add_argument("--test", type=int, default=60)
    o.add_argument("--score", default="sharpe", choices=["sharpe", "sortino", "calmar"])
    add_common(o)
    o.add_argument("--json", action="store_true")
    o.set_defaults(func=cmd_optimize)

    pp = sub.add_parser("paper", help="paper-trade a strategy against live data")
    pp.add_argument("--strategy", required=True)
    pp.add_argument("--params", default=None)
    pp.add_argument("--json", action="store_true")
    add_common(pp)
    pp.set_defaults(func=cmd_paper)

    sub.add_parser("live-status", help="report what going live would require").set_defaults(
        func=cmd_live_status
    )

    a = sub.add_parser("advise", help="ask the analysis layer for a recommendation")
    a.add_argument("--advisor", default="local", choices=["local", "hermes"])
    a.add_argument("--question", default=None)
    add_common(a)
    a.set_defaults(func=cmd_advise)

    f = sub.add_parser("firm", help="run the multi-agent trading firm")
    f.add_argument("--advisor", default="local", choices=["local", "hermes", "none"])
    f.add_argument("--kind", default="crypto", choices=["crypto", "stock"])
    f.add_argument("--json", action="store_true")
    add_common(f)
    f.set_defaults(func=cmd_firm)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {}
    if getattr(args, "data_dir", None):
        overrides["cache_db"] = f"{args.data_dir}/cache.db"
        overrides["state_db"] = f"{args.data_dir}/state.db"
    if getattr(args, "cache_db", None):
        overrides["cache_db"] = args.cache_db
    args.config = load_config(
        request_min_interval=2.0,
        cache_ttl_seconds=1800,
        **overrides,
    )
    try:
        return args.func(args)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
