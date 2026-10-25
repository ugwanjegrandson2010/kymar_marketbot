# MarketBot

An algorithmic trading system: backtest, paper trade, and — only behind an explicit
opt-in — trade live. Covers crypto (CoinGecko) and stocks/ETFs (Yahoo Finance).
Standard library only: no pandas, no numpy, no yfinance, no install step.

```
python -m marketbot backtest --symbol BTC-USD --strategy sma_crossover
python -m marketbot compare --symbol ETH-USD
python -m marketbot firm --symbol BTC-USD
python -m marketbot-tui
python -m marketbot.api            # HTTP + dashboard on :8000
```

---

## The one thing to know before anything else

**Live trading cannot be reached by accident.** It requires all three of:

1. `MARKETBOT_LIVE_TRADING=1` in the environment
2. an exchange name and API credentials
3. the literal confirmation phrase: `I understand this trades real money`

`broker/live.py` refuses to construct without all three, and `load_broker()`
returns a `PaperBroker` on every other path. There is no code path that enables
live trading implicitly, and no HTTP endpoint that places a live order.

**This is educational software. Nothing here is financial advice.** Backtested
results are not predictions. Most strategies in this repo lose money.

---

## What is here

| Area | Contents |
| --- | --- |
| `data/` | CoinGecko provider (all ~22k coins, paginated, demo-key aware), Yahoo provider (stocks, ETFs, crypto majors — full OHLCV), SQLite candle cache with TTL, universe builder with a liquidity floor |
| `indicators/` | 55 indicators across trend, momentum, volatility, volume, and candlestick patterns, plus a name registry |
| `strategies/` | 28 strategies: classic indicator systems, 12 attributed to named traders, and a composite strategy that votes across every indicator |
| `backtest/` | Event-driven engine with fees, slippage, and next-bar execution; metrics; walk-forward optimizer with parameter-stability analysis |
| `risk/` | Position sizing from stop distance (not conviction), quarter-Kelly, volatility targeting, and hard drawdown / daily-loss kill switches |
| `broker/` | Paper broker and a locked live broker behind a shared interface |
| `ai/` | Pluggable advisor (local rules or Hermes CLI) plus a multi-agent firm: analysts → bull/bear debate → trader → risk veto → portfolio approval |
| `tui.py` | Curses terminal UI: Markets, Backtest, Compare, Optimize, AI Firm, Indicators, Risk |
| `api.py` + `dashboard/` | FastAPI service and a dark web dashboard |

## Design decisions that matter

**No look-ahead.** A signal generated at bar *i* is filled at bar *i+1*'s open.
The strategy only ever receives `candles[:i+1]`. Every indicator returns a list
aligned to its input with `None` in warm-up positions, and
`scripts/verify_indicators.py` checks that truncating the future leaves past
values unchanged.

**Breakouts compare against prior bars.** A Donchian channel computed through bar
*i* contains bar *i*'s own high, so `close > channel_high` can never fire. Every
breakout strategy compares against the channel at *i-1*. This is the kind of bug
that produces a perfect-looking backtest with zero trades; `test_backtest.py`
now asserts that strategies actually trade.

**Fees and slippage are real.** 0.1% per side plus 0.05% adverse slippage by
default. A test asserts that a non-zero fee produces a different result from a
zero fee, so the cost model cannot silently vanish.

**Risk is external to strategy.** Position size comes from `risk_per_trade` and
the stop distance, never from how confident the signal feels. A breached
drawdown or daily-loss limit halts trading until explicitly reset, and the risk
manager's veto cannot be overridden by the portfolio manager.

**CoinGecko's limits are respected, not hidden.** Its `market_chart` endpoint
returns close prices only — no high/low — so bars from it carry `high == low`
and cannot trigger intrabar stops. Yahoo is the primary source for coins it
covers, because it supplies real OHLCV. The volume floor in the universe
builder drops long-tail tokens that cannot be traded at size.

## Quick start

```bash
cd ~/market-bot

# no install needed
python -m marketbot indicators --list

# backtest one strategy
python -m marketbot backtest --symbol BTC-USD --strategy turtle_breakout

# rank every strategy on one symbol
python -m marketbot compare --symbol ETH-USD

# walk-forward optimize (train/test split, out-of-sample scoring)
python -m marketbot optimize --strategy sma_crossover --symbol BTC-USD

# the multi-agent firm
python -m marketbot firm --symbol BTC-USD

# terminal UI
python -m marketbot.tui_main

# web dashboard + API
python -m marketbot.api          # then open http://127.0.0.1:8000
```

## Configuration

Every setting is overridable by a `MARKETBOT_*` environment variable. The
important ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MARKETBOT_LIVE_TRADING` | `false` | Must be explicitly on for live trading |
| `MARKETBOT_COINGECKO_API_KEY` | — | Demo API key; sent as `x-cg-demo-api-key` |
| `MARKETBOT_EXCHANGE` / `..._API_KEY` / `..._SECRET` | — | Exchange credentials |
| `MARKETBOT_STARTING_CASH` | `10000` | Paper/live account size |
| `MARKETBOT_FEE_RATE` | `0.001` | Per side |
| `MARKETBOT_SLIPPAGE_RATE` | `0.0005` | Per side, adverse |
| `MARKETBOT_RISK_PER_TRADE` | `0.01` | Fraction of equity risked per trade |
| `MARKETBOT_MAX_DRAWDOWN_PCT` | `0.20` | Halt trading below this drawdown |
| `MARKETBOT_MAX_DAILY_LOSS_PCT` | `0.03` | Halt for the day below this loss |
| `MARKETBOT_REQUEST_MIN_INTERVAL` | `2.0` | Seconds between API calls |

Copy `.env.example` to `.env` to set these persistently. The key is never logged,
never written to results, and never sent anywhere except the exchange it belongs
to.

## The AI layer

Two ways to plug a model in.

`--advisor hermes` drives the Hermes agent CLI with a structured prompt and
parses a structured reply, so the model chooses a strategy and parameters while
the deterministic layer keeps control of sizing and risk.

`firm` runs the multi-agent pipeline modelled on Tauric Research's TradingAgents:
five analysts (technical, momentum, volatility, sentiment, fundamental) produce
views, a bull/bear debate reconciles them, the trader decides, and the risk
manager holds a veto the portfolio manager cannot override. The sentiment role
is explicit that it uses price/volume as a proxy when no news feed is configured.

The model never places an order. It returns an `Advice` object; the caller
decides whether to act, and risk limits sit outside its reach.

## Tests

```bash
python -m unittest discover -s tests -v      # 140+ tests
python scripts/verify_indicators.py           # indicator integrity on live data
```

Coverage: indicators (alignment, no-look-ahead, hand-computed values), risk
sizing and kill switches, every strategy running end-to-end, engine mechanics
(fees, next-bar fills, drawdown halts), the data layer and cache, the AI layer
and the risk veto, and the TUI driven through a pseudo-terminal.

Network tests skip when the API is unreachable, so the suite runs offline.
