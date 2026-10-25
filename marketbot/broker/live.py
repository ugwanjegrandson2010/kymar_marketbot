"""Live exchange trading adapter. Locked by design.

Nothing in this module can be reached accidentally. A live broker can only be
constructed when ALL of the following are true:

  1. MARKETBOT_LIVE_TRADING is explicitly set to 1/true/yes/on;
  2. an exchange name and a non-empty API key and secret are present;
  3. a human confirmation phrase is supplied, matching the expected string.

The confirmation phrase is the guard against a script or a scheduled job turning
live trading on without a person deciding to. Omit it and construction raises.

There is no code path that enables live trading implicitly: `load_broker()`
returns a PaperBroker unless `live_ok()` is true, and `live_ok()` cannot be
satisfied without the environment variable.

Credentials are read from the environment or a `.env` file and never written to
logs, results, or the state database.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..config import Config
from ..model import Candle, Order, OrderStatus
from .base import Broker, BrokerError, Fill

CONFIRMATION_PHRASE = "I understand this trades real money"


def _read_env_file(path: Path) -> dict:
    """Parse a minimal KEY=VALUE .env file. Never raises."""
    out = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return out


def credentials(exchange: Optional[str] = None) -> dict:
    """Read exchange credentials without ever printing them."""
    env = dict(os.environ)
    env.update(_read_env_file(Path.cwd() / ".env"))
    exchange = (exchange or env.get("MARKETBOT_EXCHANGE", "")).lower()
    prefix = f"{exchange.upper()}_" if exchange else "MARKETBOT_"
    return {
        "exchange": exchange,
        "key": env.get(f"{prefix}API_KEY") or env.get("MARKETBOT_API_KEY") or "",
        "secret": env.get(f"{prefix}SECRET") or env.get("MARKETBOT_SECRET") or "",
        "passphrase": env.get(f"{prefix}PASSPHRASE") or env.get("MARKETBOT_PASSPHRASE") or "",
        "sandbox": str(
            env.get(f"{prefix}SANDBOX", env.get("MARKETBOT_SANDBOX", "false"))
        ).lower() in ("1", "true", "yes", "on"),
    }


def live_ok(confirmation: Optional[str] = None) -> tuple[bool, str]:
    """Report whether live trading may be enabled, and why not if it may not.

    Returns (allowed, reason). The reason string is always populated on refusal
    so the operator sees exactly which condition failed.
    """
    flag = str(os.environ.get("MARKETBOT_LIVE_TRADING", "")).lower()
    if flag not in ("1", "true", "yes", "on"):
        return False, "MARKETBOT_LIVE_TRADING is not enabled (refusing live trading)"
    creds = credentials()
    if not creds["exchange"]:
        return False, "no exchange configured (set MARKETBOT_EXCHANGE)"
    if not creds["key"] or not creds["secret"]:
        return False, f"no API credentials for exchange {creds['exchange']!r}"
    if confirmation != CONFIRMATION_PHRASE:
        return False, (
            "live trading requires the explicit confirmation phrase; refusing to proceed"
        )
    return True, "live trading enabled"


class LiveBroker(Broker):
    """Live order execution via ccxt.

    Construction is the gate: `live_ok()` must pass, which requires the
    environment flag, credentials, and the confirmation phrase. A missing ccxt
    installation is also fatal rather than silently falling back to simulation.
    """

    name = "live"

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        confirmation: Optional[str] = None,
        exchange_name: Optional[str] = None,
    ):
        cfg = config or Config()
        allowed, reason = live_ok(confirmation)
        if not allowed:
            raise BrokerError(f"refusing to construct live broker: {reason}")

        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise BrokerError(
                "ccxt is required for live trading but is not installed"
            ) from e

        creds = credentials(exchange_name)
        exchange_cls = getattr(ccxt, creds["exchange"], None)
        if exchange_cls is None:
            raise BrokerError(f"ccxt has no exchange named {creds['exchange']!r}")

        self.exchange = exchange_cls({
            "apiKey": creds["key"],
            "secret": creds["secret"],
            "password": creds["passphrase"] or None,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.sandbox = creds["sandbox"]
        if self.sandbox:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass

        super().__init__(
            cash=0.0,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            allow_short=cfg.allow_short,
        )
        self._prices: dict[str, float] = {}
        self._live_mode = True

    # -- Broker interface -------------------------------------------------
    def get_price(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol)

    def submit(self, order: Order) -> Order:
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "non-positive quantity"
            return order
        market = self._market_for(order.symbol)
        side = "sell" if order.side.value == "sell" else "buy"
        try:
            if order.type.value == "market":
                filled = self.exchange.create_order(market, "market", side, order.quantity)
            else:
                price = order.limit_price or order.stop_price
                kind = "limit" if order.type.value == "limit" else "market"
                filled = self.exchange.create_order(market, kind, side, order.quantity, price)
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"exchange error: {e}"
            self.orders.append(order)
            return order
        order.filled_qty = float(filled.get("amount") or order.quantity)
        order.filled_price = float(filled.get("price") or 0.0)
        order.fee = float(filled.get("fee", {}).get("cost", 0.0) or 0.0)
        order.status = OrderStatus.FILLED
        self.orders.append(order)
        return order

    def tick(self, candle: Candle, bar_index: int = 0) -> list[Fill]:
        """Fetch the live ticker and update marks. Live fills happen in submit()."""
        self._prices[candle.symbol] = candle.close
        return []

    def _market_for(self, symbol: str) -> str:
        """Convert a plain symbol like BTC-USD into an exchange market id."""
        return symbol.replace("-USD", "/USD").replace("-", "/")

    def balance(self) -> dict:
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            raise BrokerError(f"could not fetch balance: {e}") from e


def load_broker(config: Optional[Config] = None, *, confirmation: Optional[str] = None) -> Broker:
    """Return a live broker only when live trading is explicitly enabled.

    Every other path returns a paper broker, so a caller that forgets to check
    still cannot reach real money.
    """
    cfg = config or Config()
    allowed, _ = live_ok(confirmation)
    if allowed:
        return LiveBroker(cfg, confirmation=confirmation)
    from .paper import PaperBroker

    return PaperBroker(
        cash=cfg.starting_cash,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        allow_short=cfg.allow_short,
    )


def describe_live_status() -> dict:
    """What would be needed to go live, without revealing any secret."""
    allowed, reason = live_ok()
    creds = credentials()
    return {
        "live_enabled": allowed,
        "reason": reason,
        "exchange": creds["exchange"] or None,
        "has_key": bool(creds["key"]),
        "has_secret": bool(creds["secret"]),
        "sandbox": creds["sandbox"],
        "confirmation_required": CONFIRMATION_PHRASE,
    }
