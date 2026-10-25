"""Curses TUI for MarketBot.

Two halves, one interface:

  * the code half — markets, backtests, strategy comparison, walk-forward
    optimization, indicator values. Deterministic, no model involved.
  * the AI half — the multi-agent firm: each analyst's view, the bull/bear
    debate, the trader's decision, and whether risk and portfolio approved it.

Built on curses from the standard library deliberately: it needs no install,
works over SSH, and cannot be broken by a dependency update.

Controls:
  Tab / Shift-Tab or 1-7   switch screen
  Up/Down, PgUp/PgDn      move the selection
  Enter                   run / refresh
  r                       refresh data
  /                       jump to the symbol input
  q                       quit
"""

from __future__ import annotations

import curses
import curses.textpad
import json
import threading
import time
from typing import Callable, Optional

from . import __version__
from .config import load_config

SCREENS = ["Markets", "Backtest", "Compare", "Optimize", "AI Firm", "Indicators", "Risk"]
SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
]


class DataLoader:
    """Fetches data off the curses thread so the UI never blocks on a socket.

    A stalled HTTP request inside a curses loop freezes the whole interface, so
    every fetch runs in a worker and the screen renders whatever is available.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[str] = None

    def start(self, key: str, fn: Callable) -> None:
        with self._lock:
            self._pending = key
        if self._thread and self._thread.is_alive():
            return  # a fetch is already in flight; the caller will re-request
        self._thread = threading.Thread(target=self._run, args=(key, fn), daemon=True)
        self._thread.start()

    def _run(self, key: str, fn: Callable) -> None:
        try:
            value = fn()
        except Exception as e:  # noqa: BLE001 - surfaced to the UI, never crashes it
            value = {"error": f"{type(e).__name__}: {e}"}
        with self._lock:
            self._cache[key] = value
            self._pending = None

    def get(self, key: str):
        with self._lock:
            return self._cache.get(key)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def invalidate(self, prefix: str = "") -> None:
        with self._lock:
            for key in [k for k in self._cache if k.startswith(prefix)]:
                self._cache.pop(key, None)


class TUI:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.data = DataLoader(self.config)
        self.screen = 0
        self.cursor = 0
        self.symbol = "BTC-USD"
        self.strategy = "sma_crossover"
        self.strategies: list[str] = []
        self.bars = 500
        self.status = "ready"
        self.scroll = 0
        self.input_mode = False
        self.input_buffer = ""
        self.input_prompt = ""

    # ------------------------------------------------------------------
    def run(self, stdscr) -> int:
        curses.curs_set(0)
        stdscr.timeout(200)
        curses.start_color()
        curses.use_default_colors()
        self._init_colors()
        self._load_strategies()
        self._request("market", self._fetch_market)
        self._request("strategies", self._fetch_strategies)

        while True:
            self._draw(stdscr)
            try:
                key = stdscr.getch()
            except curses.error:
                continue
            if key == -1:
                continue
            action = self._handle_key(key)
            if action == "quit":
                return 0

    def _init_colors(self) -> None:
        try:
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)
        except curses.error:
            pass

    # -- data ----------------------------------------------------------
    def _load_strategies(self) -> None:
        try:
            from .strategies import available

            self.strategies = available()
        except Exception:
            self.strategies = []

    def _fetch_market(self):
        from .data import Universe

        with Universe(self.config) as u:
            candles = u.yahoo.fetch_candles(self.symbol, interval="1d", limit=self.bars)
            history = u.cache.coverage("yahoo", self.symbol, "1d")
        return {"candles": candles, "coverage": history}

    def _fetch_strategies(self):
        from .strategies import available, describe

        return [{"name": n, "description": describe(n)["description"]} for n in available()]

    def _request(self, key: str, fn: Callable) -> None:
        self.data.start(key, fn)
        self.status = f"loading {key}..."

    # -- input ---------------------------------------------------------
    def _handle_key(self, key: int) -> Optional[str]:
        if self.input_mode:
            return self._handle_input(key)

        if key in (ord("q"), ord("Q")):
            return "quit"
        if key in (ord("\t"), curses.KEY_RIGHT):
            self.screen = (self.screen + 1) % len(SCREENS)
            self.cursor = 0
            self.scroll = 0
        elif key in (curses.KEY_BTAB, curses.KEY_LEFT):
            self.screen = (self.screen - 1) % len(SCREENS)
            self.cursor = 0
            self.scroll = 0
        elif ord("1") <= key <= ord("7"):
            self.screen = key - ord("1")
            self.cursor = 0
            self.scroll = 0
        elif key in (curses.KEY_UP, ord("k")):
            self.cursor = max(0, self.cursor - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            self.cursor += 1
        elif key == curses.KEY_PPAGE:
            self.cursor = max(0, self.cursor - 10)
        elif key == curses.KEY_NPAGE:
            self.cursor += 10
        elif key in (curses.KEY_HOME,):
            self.cursor = 0
        elif key == ord("r"):
            self._refresh()
        elif key == ord("/"):
            self._begin_input("symbol", self.symbol)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            self._activate()
        return None

    def _begin_input(self, prompt: str, initial: str) -> None:
        self.input_mode = True
        self.input_prompt = prompt
        self.input_buffer = initial

    def _handle_input(self, key: int) -> Optional[str]:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            value = self.input_buffer.strip()
            if self.input_prompt == "symbol" and value:
                self.symbol = value.upper()
                self.data.invalidate("market")
                self._request("market", self._fetch_market)
            self.input_mode = False
            self.input_buffer = ""
            return None
        if key == 27:  # Escape
            self.input_mode = False
            self.input_buffer = ""
            return None
        if key in (curses.KEY_BACKSPACE, 127, 8):
            self.input_buffer = self.input_buffer[:-1]
            return None
        if 32 <= key < 127:
            self.input_buffer += chr(key)
        return None

    def _activate(self) -> None:
        name = SCREENS[self.screen]
        if name == "Markets":
            self.symbol = SYMBOLS[self.cursor % len(SYMBOLS)]
            self.data.invalidate("market")
            self._request("market", self._fetch_market)
        elif name == "Backtest":
            self._run_backtest()
        elif name == "Compare":
            self._run_compare()
        elif name == "Optimize":
            self._run_optimize()
        elif name == "AI Firm":
            self._run_firm()
        elif name == "Indicators":
            self._run_indicators()

    def _refresh(self) -> None:
        self.data.invalidate()
        self._request("market", self._fetch_market)
        self._request("strategies", self._fetch_strategies)

    # -- data actions --------------------------------------------------
    def _run_backtest(self) -> None:
        if not self.strategies:
            self.status = "no strategies loaded"
            return
        self.strategy = self.strategies[self.cursor % len(self.strategies)]
        from .backtest import run_backtest
        from .data import Universe
        from .strategies import get

        def work():
            with Universe(self.config) as u:
                candles = u.yahoo.fetch_candles(self.symbol, interval="1d", limit=self.bars)
            result = run_backtest(self.symbol, candles, get(self.strategy), self.config)
            return result.to_dict()

        self._request("backtest", work)

    def _run_compare(self) -> None:
        from .backtest.walkforward import compare_strategies
        from .data import Universe

        def work():
            with Universe(self.config) as u:
                candles = u.yahoo.fetch_candles(self.symbol, interval="1d", limit=self.bars)
            return compare_strategies(self.symbol, candles, self.strategies or None,
                                      self.config)

        self._request("compare", work)

    def _run_optimize(self) -> None:
        if not self.strategies:
            self.status = "no strategies loaded"
            return
        self.strategy = self.strategies[self.cursor % len(self.strategies)]
        from .backtest.walkforward import parameter_stability, walk_forward
        from .data import Universe

        def work():
            with Universe(self.config) as u:
                candles = u.yahoo.fetch_candles(self.symbol, interval="1d", limit=max(self.bars, 1200))
            grid = {"period": [10, 20, 50]}
            wf = walk_forward(self.symbol, candles, self.strategy, grid, self.config,
                              train_bars=250, test_bars=60)
            return {**wf.to_dict(), "stability": parameter_stability(wf)}

        self._request("optimize", work)

    def _run_firm(self) -> None:
        from .ai import AnalysisRequest, make_firm
        from .data import Universe

        def work():
            with Universe(self.config) as u:
                candles = u.yahoo.fetch_candles(self.symbol, interval="1d", limit=self.bars)
            req = AnalysisRequest(
                symbol=self.symbol,
                candles=candles,
                strategy_options=self.strategies or ["composite"],
                constraints={"kind": "crypto" if self.symbol.endswith("USD") else "stock"},
            )
            firm = make_firm(self.config, advisor_name="local")
            return firm.run(req)

        self._request("firm", work)

    def _run_indicators(self) -> None:
        from .data import Universe
        from .indicators import available, compute

        def work():
            with Universe(self.config) as u:
                candles = u.yahoo.fetch_candles(self.symbol, interval="1d", limit=self.bars)
            out = {}
            for name in available():
                try:
                    series = compute(name, candles, 14)
                    defined = [v for v in series if v is not None]
                    out[name] = {
                        "defined": len(defined),
                        "last": round(series[-1], 6) if series[-1] is not None else None,
                    }
                except Exception:  # noqa: BLE001
                    out[name] = {"defined": 0, "last": None}
            return out

        self._request("indicators", work)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _draw(self, stdscr) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 8 or width < 40:
            stdscr.addstr(0, 0, "terminal too small (need >= 40x8)")
            stdscr.refresh()
            return

        self._draw_header(stdscr, width)
        body_top = 3
        body_height = height - body_top - 3
        try:
            handler = {
                "Markets": self._draw_markets,
                "Backtest": self._draw_backtest,
                "Compare": self._draw_compare,
                "Optimize": self._draw_optimize,
                "AI Firm": self._draw_firm,
                "Indicators": self._draw_indicators,
                "Risk": self._draw_risk,
            }[SCREENS[self.screen]]
            handler(stdscr, body_top, body_height, width)
        except Exception as e:  # noqa: BLE001 - never let a render bug kill the UI
            self._safe(stdscr, body_top, 0, f"render error: {e}", width)

        self._draw_footer(stdscr, height, width)
        if self.input_mode:
            self._draw_input(stdscr, height, width)
        stdscr.refresh()

    def _safe(self, stdscr, y: int, x: int, text: str, width: int) -> None:
        """Write truncated to the screen width; curses raises on overflow."""
        if y >= stdscr.getmaxyx()[0] - 1:
            return
        stdscr.addnstr(y, x, text, max(0, width - x - 1))

    def _draw_header(self, stdscr, width: int) -> None:
        title = f" MarketBot {__version__} "
        stdscr.addstr(0, 0, title.ljust(width), curses.color_pair(6))
        tabs = " ".join(
            f"{i+1}:{name}" for i, name in enumerate(SCREENS)
        )
        line = f" {self.symbol}  |  {tabs} "
        self._safe(stdscr, 1, 0, line[: width - 1], width)
        active = SCREENS[self.screen]
        self._safe(stdscr, 2, 0, f" {active}  —  {self.status}"[: width - 1], width)

    def _draw_footer(self, stdscr, height: int, width: int) -> None:
        help_line = " Tab:screen  Up/Down:select  Enter:run  /:symbol  r:refresh  q:quit "
        if self.data.busy:
            help_line += "[loading...]"
        self._safe(stdscr, height - 2, 0, help_line[: width - 1], width)
        from .broker.live import describe_live_status

        try:
            status = describe_live_status()
            mode = "LIVE" if status["live_enabled"] else "PAPER"
            color = 3 if status["live_enabled"] else 2
            text = f" mode: {mode}  cash: {self.config.starting_cash:,.0f}  risk/trade: {self.config.risk_per_trade:.1%} "
            stdscr.addstr(height - 1, 0, text[: width - 1], curses.color_pair(color))
        except Exception:  # noqa: BLE001
            pass

    def _draw_input(self, stdscr, height: int, width: int) -> None:
        prompt = f" {self.input_prompt}> {self.input_buffer}"
        stdscr.addstr(height - 3, 0, prompt[: width - 1], curses.color_pair(4))
        curses.curs_set(1)
        try:
            stdscr.move(height - 3, min(len(prompt), width - 1))
        except curses.error:
            pass

    # -- screens -------------------------------------------------------
    def _draw_markets(self, stdscr, top: int, height: int, width: int) -> None:
        data = self.data.get("market")
        y = top
        if data is None:
            self._safe(stdscr, y, 0, "loading market data...", width)
            return
        if "error" in data:
            self._safe(stdscr, y, 0, f"error: {data['error']}", width)
            return
        candles = data.get("candles") or []
        if candles:
            last = candles[-1]
            first = candles[0]
            change = (last.close / first.close - 1) * 100
            cov = data.get("coverage") or {}
            lines = [
                f"symbol      {self.symbol}",
                f"bars        {len(candles)}",
                f"last close  {last.close:,.4f}",
                f"window chg  {change:+.2f}%",
                f"high        {max(c.high for c in candles):,.4f}",
                f"low         {min(c.low for c in candles):,.4f}",
                f"cache bars  {cov.get('bars', 'n/a')}",
            ]
            for line in lines:
                self._safe(stdscr, y, 0, line, width)
                y += 1
        y += 1
        self._safe(stdscr, y, 0, "quick symbols:", width)
        y += 1
        for i, sym in enumerate(SYMBOLS):
            marker = ">" if sym == self.symbol else " "
            self._safe(stdscr, y, 0, f"{marker} {sym}", width)
            y += 1
            if y >= top + height:
                break

    def _draw_backtest(self, stdscr, top: int, height: int, width: int) -> None:
        y = top
        self._safe(stdscr, y, 0, f"strategy: {self.strategy}   (Up/Down to choose, Enter to run)", width)
        y += 1
        data = self.data.get("backtest")
        if data is None:
            self._safe(stdscr, y, 0, "press Enter to run the backtest", width)
            return
        if "error" in data:
            self._safe(stdscr, y, 0, f"error: {data['error']}", width)
            return
        m = data.get("metrics") or {}
        rows = [
            f"return        {m.get('total_return', 0):>10.1%}",
            f"annualized    {m.get('annualized_return') if m.get('annualized_return') is not None else 0:>10.1%}",
            f"sharpe        {m.get('sharpe') if m.get('sharpe') is not None else 0:>10.2f}",
            f"max drawdown  {m.get('max_drawdown', 0):>10.1%}",
            f"halted        {str(data.get('halted')):>10} {data.get('halt_reason', '')}",
        ]
        for line in rows:
            self._safe(stdscr, y, 0, line, width)
            y += 1
        y += 1
        ts = m.get("trade_stats") or {}
        self._safe(stdscr, y, 0, "trade stats", width)
        y += 1
        for key in ("trades", "win_rate", "profit_factor", "net_pnl", "expectancy", "avg_bars_held"):
            value = ts.get(key)
            if isinstance(value, float):
                value = f"{value:,.3f}"
            self._safe(stdscr, y, 0, f"  {key:<16} {value}", width)
            y += 1
        y += 1
        trades = data.get("trades") or []
        self._safe(stdscr, y, 0, f"trades ({len(trades)})", width)
        y += 1
        for t in trades[-height + y - top if height > y - top else -10:]:
            pnl = t.get("pnl", 0)
            sign = "+" if pnl > 0 else ""
            self._safe(
                stdscr, y, 0,
                f"  {t.get('side', ''):5} {t.get('entry_price', 0):>12.4f} -> "
                f"{t.get('exit_price', 0):>12.4f}  {sign}{pnl:>10.2f}  "
                f"{t.get('exit_reason', '')}",
                width,
            )
            y += 1
            if y >= top + height:
                break

    def _draw_compare(self, stdscr, top: int, height: int, width: int) -> None:
        y = top
        self._safe(stdscr, y, 0, "Enter to rank every strategy on this symbol", width)
        y += 1
        data = self.data.get("compare")
        if data is None:
            return
        if "error" in data:
            self._safe(stdscr, y, 0, f"error: {data['error']}", width)
            return
        header = f"  {'strategy':<26} {'return':>9} {'sharpe':>7} {'maxDD':>8} {'trades':>7} {'win':>6}"
        self._safe(stdscr, y, 0, header, width)
        y += 1
        for row in data:
            if "error" in row:
                self._safe(stdscr, y, 0, f"  {row['strategy']:<26} error", width)
                y += 1
                continue
            wr = row.get("win_rate")
            wr_s = f"{wr:.0%}" if isinstance(wr, float) else "-"
            sh = row.get("sharpe")
            sh_s = f"{sh:.2f}" if isinstance(sh, float) else "-"
            line = (
                f"  {row['strategy']:<26} {row.get('total_return', 0):>8.1%} "
                f"{sh_s:>7} {row.get('max_drawdown', 0):>7.1%} "
                f"{row.get('trades', 0):>7} {wr_s:>6}"
            )
            self._safe(stdscr, y, 0, line, width)
            y += 1
            if y >= top + height:
                break

    def _draw_optimize(self, stdscr, top: int, height: int, width: int) -> None:
        y = top
        self._safe(stdscr, y, 0, f"strategy: {self.strategy}   (walk-forward: train 250 / test 60)", width)
        y += 1
        data = self.data.get("optimize")
        if data is None:
            self._safe(stdscr, y, 0, "press Enter to run walk-forward optimization", width)
            return
        if "error" in data:
            self._safe(stdscr, y, 0, f"error: {data['error']}", width)
            return
        for line in (
            f"windows           {len(data.get('windows', []))}",
            f"out-of-sample ret {data.get('oos_return')}",
            f"out-of-sample shp {data.get('oos_sharpe')}",
            f"consistency       {data.get('consistency', 0):.0%}",
            f"total trades      {data.get('total_trades')}",
            f"verdict           {data.get('verdict')}",
            f"param stability   {data.get('stability', {}).get('stable')}",
        ):
            self._safe(stdscr, y, 0, line, width)
            y += 1
        y += 1
        self._safe(stdscr, y, 0, "per-window (out-of-sample)", width)
        y += 1
        for w in data.get("windows", []):
            test = w.get("test") or {}
            flag = " degraded" if w.get("degraded") else ""
            self._safe(
                stdscr, y, 0,
                f"  test {w['test'][0]:>5}-{w['test'][1]:<5} "
                f"ret {test.get('total_return', 0):>8.1%}  "
                f"trades {test.get('trade_stats', {}).get('trades', 0):>3}{flag}",
                width,
            )
            y += 1
            if y >= top + height:
                break

    def _draw_firm(self, stdscr, top: int, height: int, width: int) -> None:
        y = top
        self._safe(stdscr, y, 0, "multi-agent firm — analysts, debate, decision, risk veto", width)
        y += 1
        data = self.data.get("firm")
        if data is None:
            self._safe(stdscr, y, 0, "press Enter to run the firm", width)
            return
        if "error" in data:
            self._safe(stdscr, y, 0, f"error: {data['error']}", width)
            return

        verdict = data.get("verdict", "")
        color = 3 if verdict.startswith("no trade") else 2
        self._safe(stdscr, y, 0, f"verdict: {verdict}", width)
        y += 2

        self._safe(stdscr, y, 0, "analysts", width)
        y += 1
        for a in data.get("analysts", []):
            conf = a.get("confidence", 0)
            d = a.get("direction", "hold")
            d_color = 2 if d == "buy" else 3 if d == "sell" else 4
            line = f"  {a.get('role', ''):<14} {d:<5} {conf:>5.0%}  {a.get('reasoning', '')}"
            stdscr.addnstr(y, 0, line[: width - 1], max(0, width - 1),
                           curses.color_pair(d_color))
            y += 1
            if y >= top + height:
                return

        y += 1
        debate = data.get("debate") or {}
        self._safe(stdscr, y, 0, "debate", width)
        y += 1
        for key in ("bull_case", "bear_case", "winner", "reconciled_score", "confidence"):
            value = debate.get(key)
            if isinstance(value, float):
                value = f"{value:+.3f}"
            self._safe(stdscr, y, 0, f"  {key:<16} {value}", width)
            y += 1
            if y >= top + height:
                return

        y += 1
        decision = data.get("decision") or {}
        self._safe(stdscr, y, 0, "decision", width)
        y += 1
        risk_ok = decision.get("risk_approved")
        pm_ok = decision.get("portfolio_approved")
        for key in ("action", "strategy", "conviction", "risk_approved", "portfolio_approved",
                    "executable", "reject_reason"):
            value = decision.get(key)
            if isinstance(value, float):
                value = f"{value:.2f}"
            line = f"  {key:<18} {value}"
            line_color = 2 if (key == "risk_approved" and value) or (key == "executable" and value) else 3
            stdscr.addnstr(y, 0, line[: width - 1], max(0, width - 1),
                           curses.color_pair(line_color))
            y += 1
            if y >= top + height:
                return

    def _draw_indicators(self, stdscr, top: int, height: int, width: int) -> None:
        y = top
        self._safe(stdscr, y, 0, f"symbol: {self.symbol}   period 14   Enter to compute", width)
        y += 1
        data = self.data.get("indicators")
        if data is None:
            return
        if "error" in data:
            self._safe(stdscr, y, 0, f"error: {data['error']}", width)
            return
        header = f"  {'indicator':<16} {'defined':>8} {'last':>14}"
        self._safe(stdscr, y, 0, header, width)
        y += 1
        items = sorted(data.items(), key=lambda kv: (kv[1].get("last") is None,
                                                     str(kv[1].get("last"))))
        for name, info in items:
            last = info.get("last")
            last_s = f"{last:,.6f}" if isinstance(last, float) else "-"
            self._safe(stdscr, y, 0, f"  {name:<16} {info.get('defined', 0):>8} {last_s:>14}", width)
            y += 1
            if y >= top + height:
                break

    def _draw_risk(self, stdscr, top: int, height: int, width: int) -> None:
        cfg = self.config
        from .broker.live import describe_live_status

        lines = [
            ("execution mode", "LIVE" if cfg.live_trading else "PAPER (simulated)"),
            ("starting cash", f"{cfg.starting_cash:,.2f}"),
            ("fee rate", f"{cfg.fee_rate:.3%} per side"),
            ("slippage", f"{cfg.slippage_rate:.3%} per side"),
            ("risk per trade", f"{cfg.risk_per_trade:.2%} of equity"),
            ("max position", f"{cfg.max_position_pct:.0%} of equity"),
            ("max open positions", str(cfg.max_open_positions)),
            ("max drawdown halt", f"{cfg.max_drawdown_pct:.0%}"),
            ("max daily loss halt", f"{cfg.max_daily_loss_pct:.0%}"),
            ("shorting", "enabled" if cfg.allow_short else "disabled"),
        ]
        y = top
        self._safe(stdscr, y, 0, "risk limits (hard stops — a breach halts trading)", width)
        y += 1
        for label, value in lines:
            self._safe(stdscr, y, 0, f"  {label:<22} {value}", width)
            y += 1
        y += 1
        try:
            status = describe_live_status()
            self._safe(stdscr, y, 0, "going live requires ALL of:", width)
            y += 1
            for req in (
                "MARKETBOT_LIVE_TRADING=1",
                "an exchange + API credentials",
                "the explicit confirmation phrase",
            ):
                self._safe(stdscr, y, 0, f"  - {req}", width)
                y += 1
            self._safe(stdscr, y, 0, f"  status: {status['reason']}", width)
            y += 1
        except Exception:  # noqa: BLE001
            pass
        y += 1
        for note in (
            "A signal is never filled on the bar it was generated from.",
            "Position size comes from the stop distance, not from conviction.",
            "The risk manager's veto cannot be overridden by the portfolio manager.",
            "Educational software. Not financial advice.",
        ):
            self._safe(stdscr, y, 0, f"  {note}", width)
            y += 1


def run_tui(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="marketbot-tui", description="Terminal UI for MarketBot")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--symbol", default="BTC-USD")
    args = p.parse_args(argv)

    overrides = {}
    if args.data_dir:
        overrides["cache_db"] = f"{args.data_dir}/cache.db"
        overrides["state_db"] = f"{args.data_dir}/state.db"
    config = load_config(**overrides)
    tui = TUI(config)
    tui.symbol = args.symbol.upper()
    try:
        return curses.wrapper(tui.run)
    except KeyboardInterrupt:
        return 0
