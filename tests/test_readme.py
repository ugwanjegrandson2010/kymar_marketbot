#!/usr/bin/env python3
"""Keep the README's counts honest.

The README states how many indicators and strategies this repo ships. Those
numbers drift silently: adding one to the registry is easy, updating the
prose is not, and nothing failed. These tests parse the table out of the
README and assert it against the registries, so the prose is now a claim the
suite enforces.

It also pins the other counts the README advertises (test count, coverage
areas), because a README that is wrong in one place stops being trustworthy in
the rest.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from marketbot import indicators, strategies

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


def _read() -> str:
    return README.read_text(encoding="utf-8")


def _table_row(text: str, prefix: str) -> str:
    """Return the README table row whose first cell is `prefix`."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"| {prefix}"):
            return stripped
    raise AssertionError(f"README has no table row starting with {prefix!r}")


def _count_from_cell(cell: str) -> int:
    """Pull the leading count out of a cell like '62 indicators across ...'."""
    match = re.match(r"\s*(\d+)\s", cell)
    assert match, f"no leading count in cell {cell!r}"
    return int(match.group(1))


class TestRegistryCounts(unittest.TestCase):
    """The README table's numbers are the registries' numbers."""

    def setUp(self):
        self.text = _read()

    def test_indicator_count_matches_the_registry(self):
        row = _table_row(self.text, "`indicators/`")
        cells = [c.strip() for c in row.strip("|").split("|")]
        # | `indicators/` | 62 indicators across ... |
        self.assertEqual(len(cells), 2, f"unexpected row shape: {row!r}")
        claimed = _count_from_cell(cells[1])
        actual = len(indicators.ALL_INDICATORS)
        self.assertEqual(
            actual, claimed,
            f"registry has {actual} indicators but the README says {claimed}",
        )

    def test_strategy_count_matches_the_registry(self):
        row = _table_row(self.text, "`strategies/`")
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(len(cells), 2, f"unexpected row shape: {row!r}")
        claimed = _count_from_cell(cells[1])
        actual = len(strategies.STRATEGIES)
        self.assertEqual(
            actual, claimed,
            f"registry has {actual} strategies but the README says {claimed}",
        )

    def test_indicator_count_is_the_close_plus_ohlc_total(self):
        """A duplicate name would inflate the table without inflating the union."""
        union = set(indicators.CLOSE_INDICATORS) | set(indicators.OHLC_INDICATORS)
        self.assertEqual(
            len(union), len(indicators.ALL_INDICATORS),
            "CLOSE and OHLC registries share a name; ALL_INDICATORS would hide it",
        )

    def test_registry_counts_are_not_suspiciously_small(self):
        """Guard the guard: a registry that emptied would satisfy a stale README."""
        self.assertGreaterEqual(len(indicators.ALL_INDICATORS), 50)
        self.assertGreaterEqual(len(strategies.STRATEGIES), 25)


class TestTestCountClaim(unittest.TestCase):
    """The README's test count is a floor, not a promise.

    The count is resolved without running a nested subprocess, because
    `python -m unittest discover -s tests` refuses to descend into a directory
    named `tests` while it is already importing modules from that same path —
    discovery doubles back on itself and reports 0 tests.
    """

    def test_claimed_count_is_never_above_the_real_suite(self):
        text = _read()
        match = re.search(r"#\s*(\d+)\s+tests", text)
        self.assertIsNotNone(match, "README no longer states a test count")
        claimed = int(match.group(1))

        loader = unittest.TestLoader()
        suite = loader.discover(str(REPO_ROOT / "tests"))
        ran = suite.countTestCases()
        self.assertGreater(
            ran, 0, "discovery found no tests; the path or pattern is wrong",
        )
        self.assertGreaterEqual(
            ran, claimed,
            f"README claims {claimed} tests but the suite has {ran}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
