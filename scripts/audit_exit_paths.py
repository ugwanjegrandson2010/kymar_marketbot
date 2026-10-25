"""Classify each strategy by which directions it can actually signal.

A strategy that can only ever emit BUY holds one position for the entire
backtest: the result looks valid but only ever tests a single entry, so it
cannot demonstrate repeated trading skill. Testing on an uptrend alone hides
this, because everything looks bullish. Both regimes are needed.
"""
import sys
import math
import random

sys.path.insert(0, ".")

from marketbot.model import Candle, SignalAction
from marketbot.strategies import available, get


def trending(n=800):
    rng = random.Random(4)
    out, p = [], 100.0
    for i in range(n):
        o = p
        c = p + 0.5 + rng.uniform(-0.4, 0.4)
        out.append(Candle(ts=1_600_000_000 + i * 86400, open=o, high=max(o, c) + 0.3,
                          low=min(o, c) - 0.3, close=c, volume=1000.0))
        p = c
    return out


def oscillating(n=800):
    """Mean-reverting series with a real open/close body per bar.

    The previous version set open == close, giving every candle a zero body, so
    no candlestick pattern could ever fire and the classification below was
    meaningless. Open is the prior close here, as in real bar data.
    """
    rng = random.Random(9)
    out = []
    prev_close = 100.0
    for i in range(n):
        price = 100.0 + 12.0 * math.sin(i / 14.0) + rng.uniform(-0.4, 0.4)
        o, c = prev_close, price
        out.append(Candle(ts=1_600_000_000 + i * 86400, open=round(o, 4),
                          high=round(max(o, c) + 0.6, 4),
                          low=round(min(o, c) - 0.6, 4),
                          close=round(c, 4), volume=1000.0))
        prev_close = c
    return out


def classify(candles, name):
    s = get(name)
    s.prepare(candles)
    seen = set()
    for i in range(5, len(candles), 2):
        seen.add(s.on_bar(i, candles[: i + 1]).action)
    return seen


print(f"{'strategy':26} {'uptrend':>18} {'oscillating':>18}  classification")
print("-" * 92)

one_directional = []
for name in available():
    up = classify(trending(), name)
    osc = classify(oscillating(), name)
    both = up | osc
    can_long = SignalAction.BUY in both
    can_short = SignalAction.SELL in both
    can_exit = SignalAction.CLOSE in both

    if not (up or osc):
        cls = "never signals"
    elif can_long and can_short:
        cls = "long + short (flips)"
    elif can_long and not can_short:
        cls = "LONG ONLY"
        one_directional.append(name)
    elif can_short and not can_long:
        cls = "SHORT ONLY"
        one_directional.append(name)
    else:
        cls = "exit only"

    fmt = lambda s: ",".join(sorted(a.value for a in s)) or "-"
    print(f"{name:26} {fmt(up):>18} {fmt(osc):>18}  {cls}")

print()
print("structurally one-directional:", one_directional or "none")
print(f"count: {len(one_directional)}")
