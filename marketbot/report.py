"""Rendered reports: numbers with the context that makes them readable.

The failure this exists to prevent is printing a bare figure. `win rate 100.0%`
tells the reader nothing; `100.0% over 1 trade — too small to mean anything`
tells them not to trust it. Every number here carries a sample size, a
baseline, or an explicit statement of what it does not mean.
"""

from __future__ import annotations

from typing import Optional, Sequence

from . import art
from .art import Palette


def _pal() -> Palette:
    """Colour only when stdout is a terminal, so piped output stays clean."""
    return art.DEFAULT


def _verdict_token(value: Optional[float], good_above: float = 0.0) -> str:
    if value is None:
        return "muted"
    return "ok" if value > good_above else "bad" if value < good_above else "muted"


def backtest_report(result, pal: Optional[Palette] = None) -> str:
    """One screen: what happened, and whether the numbers mean anything."""
    pal = pal or _pal()
    m = result.metrics
    ts = m.trade_stats
    acc = m.accuracy
    n = ts.get("trades", 0)

    # --- the return, explained -----------------------------------------
    ret_line = art.stat(
        "Total return",
        art.signed(m.total_return),
        "gain or loss over the whole window; negative is a loss",
        "ok" if m.total_return > 0 else "bad" if m.total_return < 0 else "muted",
        pal,
    )
    ann = m.annualized_return
    ann_line = art.stat(
        "Annualized",
        art.pct(ann) if ann is not None else "n/a",
        "compounded yearly"
        if ann is not None
        else "undefined — need more than one bar",
        _verdict_token(ann) if ann is not None else "muted",
        pal,
    )

    # --- risk ----------------------------------------------------------
    dd_line = art.stat(
        "Max drawdown",
        art.pct(m.max_drawdown),
        "worst peak-to-trough loss; lower is better",
        "ok" if m.max_drawdown < 0.10 else "warn" if m.max_drawdown < 0.25 else "bad",
        pal,
    )
    sharpe = m.sharpe
    sharpe_line = art.stat(
        "Sharpe",
        f"{sharpe:.2f}" if sharpe is not None else "n/a",
        "return per unit of volatility; above 1.0 is the usual bar"
        if sharpe is not None
        else "undefined — needs at least two returns",
        _verdict_token(sharpe, 1.0) if sharpe is not None else "muted",
        pal,
    )

    # --- trades, with the sample-size warning --------------------------
    if n == 0:
        trades_line = art.stat("Trades", "0", "no closed trades — no rate below is meaningful", "warn", pal)
    elif n < 20:
        trades_line = art.stat(
            "Trades", str(n), f"only {n} — every rate below is a small-sample estimate", "warn", pal
        )
    else:
        trades_line = art.stat("Trades", str(n), "enough for the rates below to be read as estimates", "ok", pal)

    wr = ts.get("win_rate")
    wr_line = art.stat(
        "Win rate",
        art.pct(wr) if wr is not None else "n/a",
        "share of trades that closed in profit" if wr is not None else "no closed trades",
        "ok" if (wr or 0) >= 0.5 else "warn",
        pal,
    )

    pf = ts.get("profit_factor")
    if pf is None:
        pf_line = art.stat("Profit factor", "n/a", "gross win / gross loss — undefined with no losing trades")
    elif pf == 0:
        pf_line = art.stat("Profit factor", "0.00", "every trade lost; gross win was zero", "bad", pal)
    else:
        pf_line = art.stat(
            "Profit factor",
            f"{pf:.2f}",
            "gross profit / gross loss; above 1.0 keeps you alive",
            "ok" if pf > 1.5 else "warn" if pf > 1.0 else "bad",
            pal,
        )

    net = ts.get("net_pnl", 0.0)
    net_line = art.stat(
        "Net P&L",
        f"{net:+,.2f}",
        "after fees; fees were "
        f"{ts.get('total_fees', 0):,.2f}",
        "ok" if net > 0 else "bad" if net < 0 else "muted",
        pal,
    )

    # --- equity curve ---------------------------------------------------
    curve = result.equity_curve
    spark = art.sparkline_labeled(curve[-200:], "equity", 44, pal) if curve else ""

    # --- halt ----------------------------------------------------------
    halt = ""
    if result.halted:
        halt = art.notice(
            "Trading halted",
            [result.halt_reason or "a hard risk limit was breached",
             "Positions were flattened. Reset the limit to resume."],
            tone="bad",
            pal=pal,
        )

    body = [
        ret_line,
        ann_line,
        dd_line,
        sharpe_line,
        "",
        trades_line,
        wr_line,
        pf_line,
        net_line,
    ]
    if spark:
        body += ["", pal.paint("Equity curve (last 200 bars)", "muted"), spark]

    out = [art.banner(result.strategy.upper(), slogan=f"{result.symbol} · {result.bars} bars", pal=pal)]
    out.append(art.panel("Performance", body, pal=pal))

    if halt:
        out += ["", halt]

    # --- recent trades as a table -------------------------------------
    trades = result.trades[-12:]
    if trades:
        rows = []
        for t in trades:
            rows.append([
                t.symbol,
                t.side.value,
                f"{t.entry_price:,.4f}",
                f"{t.exit_price:,.4f}",
                f"{t.pnl:+,.2f}",
                t.exit_reason or "",
            ])
        out += ["", pal.paint("Recent trades", "accent"), art.table(
            ["symbol", "side", "entry", "exit", "pnl", "exit reason"],
            rows, pal, align_right=[2, 3, 4],
        )]

    out += ["", art.rule(78, pal=pal)]
    out.append(pal.paint(
        "Educational software. Not financial advice. Past results do not predict future ones.",
        "muted",
    ))
    return "\n".join(out)


def compare_report(rows: Sequence[dict], pal: Optional[Palette] = None) -> str:
    """Strategy ranking, with the caveat that ranking needs a shared window."""
    pal = pal or _pal()
    if not rows:
        return art.notice("empty", ["no strategies produced a result"], pal=pal)

    table_rows = []
    for r in rows:
        if "error" in r:
            table_rows.append([r["strategy"], "error", "-", "-", "-", "-"])
            continue
        table_rows.append([
            r["strategy"],
            art.signed(r.get("total_return", 0.0), 1),
            f"{r['sharpe']:.2f}" if r.get("sharpe") is not None else "n/a",
            art.pct(r.get("max_drawdown")),
            str(r.get("trades", 0)),
            art.pct(r.get("win_rate")) if r.get("win_rate") is not None else "n/a",
        ])

    table_block = art.table(
        ["strategy", "return", "sharpe", "maxDD", "trades", "win"],
        table_rows, pal,
    )
    # A table is already a box. Nesting it inside a panel doubles every border,
    # so title it with a rule instead.
    lines = [pal.paint("Strategy ranking", "accent"), table_block]

    n_profitable = sum(1 for r in rows if (r.get("total_return") or 0) > 0)
    lines += [
        "",
        art.stat(
            "Profitable",
            f"{n_profitable}/{len(rows)}",
            "strategies with a positive return on this window",
            "ok" if n_profitable > len(rows) / 2 else "warn",
            pal,
        ),
        pal.paint(
            "Ranking on one window proves nothing: a strategy that won here may be "
            "fitting this window. Use `optimize` for out-of-sample testing.",
            "muted",
        ),
    ]
    return "\n".join(lines)


def firm_report(result: dict, pal: Optional[Palette] = None) -> str:
    """The multi-agent decision, with the risk veto shown explicitly."""
    pal = pal or _pal()
    lines = [art.banner("THE FIRM", slogan=result.get("symbol", ""), pal=pal)]

    analyst_rows = []
    for a in result.get("analysts", []):
        analyst_rows.append([
            a.get("role", ""),
            a.get("direction", ""),
            f"{a.get('confidence', 0) * 100:.0f}%",
            a.get("reasoning", ""),
        ])
    lines.append(pal.paint("Analysts", "accent"))
    lines.append(art.table(["role", "view", "conf", "reasoning"], analyst_rows, pal))

    debate = result.get("debate", {})
    lines.append(art.panel("Debate", [
        art.stat("Winner", str(debate.get("winner", "-")).upper(),
                 "which side carried more weighted evidence"),
        art.stat("Score", f"{debate.get('reconciled_score', 0):+.2f}",
                 "bull minus bear, scaled to -1..+1; near zero means no conviction"),
        art.stat("Confidence", f"{debate.get('confidence', 0) * 100:.0f}%",
                 "how much the two sides disagreed; low means they agreed"),
    ], pal=pal))

    d = result.get("decision", {})
    risk_ok = bool(d.get("risk_approved"))
    lines.append(art.panel("Decision", [
        art.stat("Action", str(d.get("action", "-")).upper(),
                 "what the trader decided"),
        art.stat("Strategy", d.get("strategy") or "—", "the system it would run"),
        art.stat("Conviction", f"{d.get('conviction', 0) * 100:.0f}%",
                 "scales position size downward, never upward"),
        art.stat("Risk approved", "YES" if risk_ok else "NO",
                 "the risk manager's veto — cannot be overridden" if risk_ok
                 else "rejected by risk management",
                 "ok" if risk_ok else "bad", pal),
        art.stat("Executable", "YES" if d.get("executable") else "NO",
                 "action + risk approval + portfolio approval, all required",
                 "ok" if d.get("executable") else "muted", pal),
    ] + ([f"Rejected: {d['reject_reason']}"] if d.get("reject_reason") else []), pal=pal))

    lines += ["", art.rule(78, pal=pal)]
    lines.append(pal.paint(f"verdict: {result.get('verdict', '')}", "accent"))
    return "\n".join(lines)


def indicators_report(data: dict, symbol: str, pal: Optional[Palette] = None) -> str:
    """Indicator values, flagged when they are undefined rather than printed as zero."""
    pal = pal or _pal()
    if not data:
        return art.notice("empty", ["no indicators computed"], pal=pal)
    rows = []
    undefined = 0
    for name in sorted(data):
        info = data[name]
        last = info.get("last")
        if last is None:
            undefined += 1
        rows.append([
            name,
            str(info.get("defined", 0)),
            f"{last:,.4f}" if isinstance(last, float) else "undefined",
        ])
    body = [art.table(["indicator", "bars", "last"], rows, pal, align_right=[1, 2])]
    if undefined:
        body += ["", pal.paint(
            f"{undefined} indicators returned no value — shown as undefined, not zero.",
            "warn",
        )]
    return "\n".join([pal.paint(f"Indicators — {symbol}", "accent")] + body)
