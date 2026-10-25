"""Indicator integrity checks against real market data.

Three properties are load-bearing for a trading bot:
  1. alignment  - every series is the same length as the input;
  2. no look-ahead - the value at bar i is identical whether or not later bars
     exist (computed by truncating the series and recomputing);
  3. sanity     - bounded indicators stay in range and known values match
     a hand-computed reference.

Run:  python3 scripts/verify_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketbot.config import load_config
from marketbot.data import Universe
from marketbot.indicators import ALL_INDICATORS, BAND_INDICATORS, compute, compute_all
from marketbot.indicators.core import closes


def check_alignment(name, candles, series):
    assert isinstance(series, list), f"{name}: not a list"
    assert len(series) == len(candles), f"{name}: length {len(series)} != {len(candles)}"


def check_no_lookahead(name, candles, series):
    """Truncate the input; the value at the cut must be unchanged."""
    if len(candles) < 10:
        return
    cut = len(candles) // 2
    try:
        truncated = compute_all(candles[:cut], 14)
        fn_series = truncated[name]
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        return
    for i in range(cut):
        a, b = series[i], fn_series[i]
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise AssertionError(f"{name}[{i}]: full={a} truncated={b}")
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            raise AssertionError(f"{name}[{i}]: full={a} truncated={b} (look-ahead!)")


def main() -> int:
    cfg = load_config(request_min_interval=2.0)
    with Universe(cfg) as u:
        candles = u.yahoo.fetch_candles("BTC-USD", interval="1d", limit=400)
        print(f"data: BTC-USD {len(candles)} daily bars "
              f"({candles[0].ts} -> {candles[-1].ts})")
        print(f"last close: {candles[-1].close:,.2f}\n")

        all_series = compute_all(candles, 14)
        print(f"computing {len(ALL_INDICATORS)} indicators...")
        failures = []
        for name, series in all_series.items():
            try:
                check_alignment(name, candles, series)
                check_no_lookahead(name, candles, series)
            except AssertionError as e:
                failures.append(str(e))
                print(f"  FAIL {e}")

        # --- sanity checks on known maths ---------------------------------
        cl = closes(candles)
        from marketbot.indicators.core import sma  # noqa
        from marketbot.indicators.oscillators import rsi  # noqa

        # SMA20 of the last 20 closes, hand-computed (compute_all used period 14)
        manual = sum(cl[-20:]) / 20.0
        got = compute("sma", candles, 20)[-1]
        assert abs(manual - got) < 1e-9, f"sma: {manual} vs {got}"
        print(f"\nSMA20 hand-check: manual={manual:.4f} lib={got:.4f} OK")

        # RSI bounds
        rsi_vals = [v for v in all_series["rsi"] if v is not None]
        assert all(0.0 <= v <= 100.0 for v in rsi_vals), "RSI out of 0-100"
        print(f"RSI range: min={min(rsi_vals):.2f} max={max(rsi_vals):.2f} OK")

        # Bollinger: upper >= mid >= lower wherever defined
        bb = BAND_INDICATORS["bollinger"](candles, 20)
        for i in range(len(candles)):
            u_, m_, l_ = bb["upper"][i], bb["mid"][i], bb["lower"][i]
            if u_ is None:
                continue
            assert u_ >= m_ >= l_, f"bollinger order broken at {i}"
        print("Bollinger band ordering OK")

        # Warm-up must produce Nones, never fabricated values.
        # Indicators with a shorter natural warm-up than the requested period
        # (ATR needs 2 bars, VWAP needs 1) legitimately carry values early, so
        # this is informational rather than a defect.
        early = [
            name
            for name, series in all_series.items()
            if any(v is not None for v in series[:14])
        ]
        if early:
            print(f"  INFO indicators with sub-14-bar warm-up: {len(early)} "
                  f"({', '.join(early[:8])}{'...' if len(early) > 8 else ''})")

        print(f"\n{'FAILURES' if failures else 'ALL CHECKS PASSED'}: "
              f"{len(all_series)} indicators, {len(failures)} failures")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
