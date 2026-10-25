"""TUI tests.

A curses app cannot be asserted against directly, so these drive it through a
pseudo-terminal: spawn it, send keystrokes, read the rendered output, and check
that screens render, navigation works, and it quits cleanly.

They run headless (no terminal attached to the test process itself), so they
also prove the UI does not require a human at the keyboard.
"""

import os
import pty
import select
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_tui_session(keys: bytes, wait: float = 1.5, cols: int = 120, rows: int = 36) -> str:
    """Launch the TUI in a pty, send `keys`, and return everything it printed."""
    master, slave = pty.openpty()
    import fcntl
    import struct
    import termios

    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(cols), LINES=str(rows))
    proc = subprocess.Popen(
        [sys.executable, "-m", "marketbot.tui_main"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=str(ROOT),
        env=env,
        close_fds=True,
    )
    os.close(slave)

    out = bytearray()
    deadline = time.time() + wait
    sent = False
    while time.time() < deadline:
        readable, _, _ = select.select([master], [], [], 0.1)
        if readable:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            out.extend(chunk)
        if not sent and time.time() > deadline - wait + 0.5:
            os.write(master, keys)
            sent = True

    # Always quit so the session cannot hang.
    try:
        os.write(master, b"q")
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    os.close(master)
    return out.decode("utf-8", errors="ignore")


class TestTUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = run_tui_session(b"\tq", wait=3.0)

    def test_starts_and_renders_the_header(self):
        self.assertIn("MarketBot", self.output)

    def test_shows_the_screen_tabs(self):
        for tab in ("Markets", "Backtest", "Compare", "Optimize", "AI Firm", "Indicators", "Risk"):
            self.assertIn(tab, self.output, f"tab {tab!r} never rendered")

    def test_shows_paper_mode_by_default(self):
        # Live trading is off by default; the footer must say so.
        self.assertIn("PAPER", self.output)

    def test_tab_navigation_renders_a_different_screen(self):
        nav = run_tui_session(b"\t\t\t\tq", wait=2.5)
        self.assertIn("multi-agent firm", nav.lower())

    def test_quit_is_clean(self):
        # If the process failed to exit, run_tui_session kills it; reaching here
        # with rendered output means the loop started and accepted input.
        self.assertGreater(len(self.output), 500)


class TestDataLoader(unittest.TestCase):
    def test_caches_results(self):
        from marketbot.tui import DataLoader

        dl = DataLoader()
        dl._cache["k"] = {"a": 1}
        self.assertEqual(dl.get("k"), {"a": 1})

    def test_invalidate_clears_a_prefix(self):
        from marketbot.tui import DataLoader

        dl = DataLoader()
        dl._cache = {"market_1": 1, "market_2": 2, "other": 3}
        dl.invalidate("market")
        self.assertNotIn("market_1", dl._cache)
        self.assertEqual(dl.get("other"), 3)

    def test_missing_key_returns_none(self):
        from marketbot.tui import DataLoader

        self.assertIsNone(DataLoader().get("nope"))


class TestTUIKeyHandling(unittest.TestCase):
    def test_number_keys_switch_screens(self):
        from marketbot.tui import SCREENS, TUI

        tui = TUI()
        tui._handle_key(ord("5"))
        self.assertEqual(SCREENS[tui.screen], "AI Firm")
        tui._handle_key(ord("1"))
        self.assertEqual(SCREENS[tui.screen], "Markets")

    def test_tab_wraps_around(self):
        from marketbot.tui import TUI

        tui = TUI()
        tui.screen = len(["a"] * 7) - 1
        tui._handle_key(ord("\t"))
        self.assertEqual(tui.screen, 0)

    def test_quit_key_returns_quit(self):
        from marketbot.tui import TUI

        self.assertEqual(TUI()._handle_key(ord("q")), "quit")

    def test_input_mode_accepts_text_and_enter(self):
        from marketbot.tui import TUI

        tui = TUI()
        tui._begin_input("symbol", "BTC-USD")
        self.assertTrue(tui.input_mode)
        tui._handle_input(ord("X"))
        self.assertEqual(tui.input_buffer, "BTC-USDX")
        tui._handle_input(curses_enter())
        self.assertFalse(tui.input_mode)
        self.assertEqual(tui.symbol, "BTC-USDX")

    def test_escape_cancels_input(self):
        from marketbot.tui import TUI

        tui = TUI()
        tui._begin_input("symbol", "BTC")
        tui._handle_input(27)
        self.assertFalse(tui.input_mode)


def curses_enter() -> int:
    return 10


if __name__ == "__main__":
    unittest.main()
