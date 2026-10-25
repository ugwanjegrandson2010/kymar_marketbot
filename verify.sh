#!/usr/bin/env bash
# Run the full verification suite for MarketBot.
# Exits non-zero if anything fails. Safe to run repeatedly.
set -uo pipefail
cd "$(dirname "$0")"

fail=0
run() {
  echo ""
  echo "=== $* ==="
  if ! "$@"; then
    echo "FAILED: $*"
    fail=1
  fi
}

echo "MarketBot verification"
echo "python: $(python3 --version 2>&1)"

run python3 -m unittest discover -s tests -v

echo ""
echo "=== indicator integrity on live data ==="
if python3 scripts/verify_indicators.py; then
  echo "indicators OK"
else
  echo "FAILED: indicator verification (network may be unavailable)"
  fail=1
fi

echo ""
echo "=== end-to-end smoke test ==="
if python3 -m marketbot backtest --symbol BTC-USD --strategy sma_crossover --bars 300; then
  echo "backtest OK"
else
  echo "FAILED: backtest"
  fail=1
fi

if python3 -m marketbot compare --symbol ETH-USD --bars 300; then
  echo "compare OK"
else
  echo "FAILED: compare"
  fail=1
fi

if python3 -m marketbot firm --symbol BTC-USD --bars 300; then
  echo "firm OK"
else
  echo "FAILED: firm"
  fail=1
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "ALL VERIFICATION PASSED"
else
  echo "VERIFICATION FAILED"
fi
exit "$fail"
