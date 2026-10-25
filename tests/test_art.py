#!/usr/bin/env python3
"""Regression tests for the art component library.

These pin the bugs found while the CLI was actually used: a table whose
headers were truncated to the width of its data, a table with no corners, a
header that was right-aligned like a value, and a table nested inside a panel
so every border appeared twice.
"""

from __future__ import annotations

import unittest

from marketbot import art


class TestTable(unittest.TestCase):
    HEADERS = ["strategy", "return", "sharpe", "maxDD", "trades", "win"]
    ROWS = [
        ["sma_crossover", "-11.9%", "-1.36", "12.8%", "1", "0.0%"],
        ["rsi_atr_breakout", "-12.3%", "-0.76", "18.2%", "2", "0.0%"],
    ]

    def setUp(self):
        # No colour, so the assertions read on the raw structure.
        self.pal = art.Palette(enabled=False)
        self.out = art.table(self.HEADERS, self.ROWS, self.pal,
                             align_right=[1, 2, 3, 4, 5])

    def test_is_a_box_with_corners(self):
        """The top and bottom must be corners, not row separators."""
        lines = self.out.split("\n")
        self.assertTrue(lines[0].startswith("┌"), f"no top-left corner: {lines[0]!r}")
        self.assertTrue(lines[0].endswith("┐"), f"no top-right corner: {lines[0]!r}")
        self.assertTrue(lines[-1].startswith("└"), f"no bottom-left corner: {lines[-1]!r}")
        self.assertTrue(lines[-1].endswith("┘"), f"no bottom-right corner: {lines[-1]!r}")

    def test_column_junctions_are_crosses(self):
        """Top/bottom use tees, the row separator uses crosses."""
        lines = self.out.split("\n")
        self.assertIn("┬", lines[0], f"top border is not a tee: {lines[0]!r}")
        self.assertIn("┴", lines[-1], f"bottom border is not a tee: {lines[-1]!r}")
        self.assertIn("┼", lines[2], f"row separator is not a cross: {lines[2]!r}")
        self.assertNotIn("┼", lines[0], "the top border must not use crosses")

    def test_headers_are_not_truncated(self):
        """'trades' is longer than every value in its column.

        Measuring only the data shrank the column to 3 cells and the header
        rendered as 'tr'.
        """
        header = self.out.split("\n")[1]
        for h in self.HEADERS:
            self.assertIn(h, header, f"header {h!r} truncated in {header!r}")

    def test_headers_are_left_aligned(self):
        header = self.out.split("\n")[1]
        # A right-aligned header shows leading spaces inside its cell.
        self.assertIn("strategy ", header)
        self.assertIn("trades ", header)

    def test_data_columns_respect_alignment(self):
        body = self.out.split("\n")[3]
        self.assertIn(" -11.9%", body)     # right-aligned numeric
        self.assertTrue(body.startswith("│sma_crossover"), "first column must be left-aligned")

    def test_every_row_is_the_same_width(self):
        widths = {len(l) for l in self.out.split("\n")}
        self.assertEqual(len(widths), 1, f"rows are not aligned: {sorted(widths)}")

    def test_empty_table_is_an_honest_notice(self):
        out = art.table(self.HEADERS, [], self.pal)
        self.assertIn("no rows", out)


class TestPanelWithTable(unittest.TestCase):
    """The nesting bug: a bordered table inside a bordered panel."""

    def test_panel_does_not_balloon(self):
        pal = art.Palette(enabled=False)
        t = art.table(["a", "b"], [["1", "2"]], pal)
        boxed = art.panel("Title", [t], pal=pal, wrap_body=False)
        widest = max(len(l) for l in boxed.split("\n"))
        self.assertLessEqual(widest, 90, f"panel ballooned to {widest} columns")

    def test_panel_width_is_the_widest_line(self):
        """A panel honours its requested width, growing only when content needs it."""
        pal = art.Palette(enabled=False)
        t = art.table(["a", "b"], [["1", "2"]], pal)
        content = max(len(l) for l in t.split("\n"))
        for requested in (40, 78):
            boxed = art.panel("Title", [t], pal=pal, wrap_body=False, width=requested)
            inner = max(requested - 4, content)
            self.assertEqual(len(boxed.split("\n")[0]), inner + 4,
                             f"panel width wrong for requested={requested}")


class TestBanner(unittest.TestCase):
    def test_rows_are_equal_width(self):
        rows = art.banner("MARKETBOT", pal=art.Palette(enabled=False)).split("\n")
        widths = {len(l) for l in rows if l}
        self.assertEqual(len(widths), 1, f"banner rows misaligned: {sorted(widths)}")

    def test_banner_is_five_rows(self):
        rows = art.banner("TEST", pal=art.Palette(enabled=False)).split("\n")
        self.assertEqual(len(rows), 5)


class TestStat(unittest.TestCase):
    def test_stat_carries_the_explanation(self):
        pal = art.Palette(enabled=False)
        line = art.stat("Trades", "0", "no closed trades — no rate below is meaningful",
                        "warn", pal)
        self.assertIn("Trades", line)
        self.assertIn("no closed trades", line)

    def test_stat_handles_none_values(self):
        pal = art.Palette(enabled=False)
        line = art.stat("Sharpe", "n/a", "undefined — need at least two returns",
                        "muted", pal)
        self.assertIn("n/a", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
