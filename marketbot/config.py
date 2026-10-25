"""Runtime configuration.

Every value can be overridden by an environment variable prefixed MARKETBOT_.
The single most important switch is LIVE_TRADING: it defaults to False and the
live broker refuses to construct unless it is explicitly enabled AND credentials
are present. There is no code path that turns live trading on implicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_PREFIX = "MARKETBOT_"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env files into os.environ.

    A `.env` file placed in the project root is read even when the bot is
    launched from another directory, and the current working directory's .env
    is read as a fallback.

    Real environment variables always win: an explicit `export` must override
    a file, not the other way round. Values are never printed, logged, or
    written into results — they are credentials.
    """
    for path in (PROJECT_ROOT / ".env", Path.cwd() / ".env"):
        try:
            if not path.is_file():
                continue
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _env(name: str, default):
    """Read MARKETBOT_<name> from the environment, else return default."""
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


DATA_DIR = Path(_env("DATA_DIR", PROJECT_ROOT / "data"))
CACHE_DB = Path(_env("CACHE_DB", DATA_DIR / "cache.db"))
STATE_DB = Path(_env("STATE_DB", DATA_DIR / "state.db"))


@dataclass
class Config:
    """All tunables in one place, env-overridable."""

    # --- execution mode -------------------------------------------------
    # False = simulated only. Never flip this without reading broker/live.py.
    live_trading: bool = False

    # --- data -----------------------------------------------------------
    cache_db: Path = CACHE_DB
    state_db: Path = STATE_DB
    cache_ttl_seconds: int = 1800
    http_timeout: int = 25
    user_agent: str = "Mozilla/5.0 (compatible; MarketBot/0.1)"
    max_retries: int = 3
    retry_backoff: float = 1.5
    # CoinGecko's free tier is IP-shared and returns 429 quickly. Stay polite,
    # and use the demo API key when one is configured for the higher limits.
    request_min_interval: float = 2.0
    coingecko_api_key: str = ""

    # --- backtest -------------------------------------------------------
    starting_cash: float = 10_000.0
    fee_rate: float = 0.001          # 0.1% taker fee, per side
    slippage_rate: float = 0.0005    # 0.05% adverse fill, per side
    allow_short: bool = True
    price_precision: int = 8

    # --- risk -----------------------------------------------------------
    risk_per_trade: float = 0.01          # 1% of equity risked per trade
    max_position_pct: float = 0.20        # never >20% of equity in one name
    max_open_positions: int = 5
    max_drawdown_pct: float = 0.20        # halt trading below -20% equity DD
    max_daily_loss_pct: float = 0.03      # halt for the day below -3% session
    kelly_fraction: float = 0.25          # quarter-Kelly, never full Kelly
    min_kelly_trades: int = 30            # no Kelly sizing on tiny samples
    # Cap a single symbol's contribution so no one coin can dominate a portfolio.
    max_symbol_weight: float = 0.35

    # --- strategies -----------------------------------------------------
    universe: tuple = field(default_factory=tuple)

    def __post_init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Mask credentials so a stray repr or log line cannot leak a secret."""
        fields = []
        for f in self.__dataclass_fields__:
            value = getattr(self, f)
            if "key" in f.lower() or "secret" in f.lower():
                value = f"<{len(str(value))} chars>" if value else "<empty>"
            fields.append(f"{f}={value!r}")
        return f"Config({', '.join(fields)})"

    def masked_summary(self) -> dict:
        """Config summary safe to log or return from an API."""
        return {
            "live_trading": self.live_trading,
            "starting_cash": self.starting_cash,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "risk_per_trade": self.risk_per_trade,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "allow_short": self.allow_short,
            "cache_db": str(self.cache_db),
            "has_coingecko_key": bool(self.coingecko_api_key),
        }


def load_config(**overrides) -> Config:
    """Build a Config from environment variables, then apply explicit overrides.

    Overrides win over the environment, which wins over defaults. This is the
    only sanctioned way to change execution mode.
    """
    cfg = Config(
        live_trading=_env("LIVE_TRADING", False),
        cache_db=Path(_env("CACHE_DB", CACHE_DB)),
        state_db=Path(_env("STATE_DB", STATE_DB)),
        cache_ttl_seconds=_env("CACHE_TTL", 1800),
        http_timeout=_env("HTTP_TIMEOUT", 25),
        request_min_interval=_env("REQUEST_MIN_INTERVAL", 2.0),
        coingecko_api_key=_env("COINGECKO_API_KEY", ""),
        starting_cash=_env("STARTING_CASH", 10_000.0),
        fee_rate=_env("FEE_RATE", 0.001),
        slippage_rate=_env("SLIPPAGE_RATE", 0.0005),
        allow_short=_env("ALLOW_SHORT", True),
        risk_per_trade=_env("RISK_PER_TRADE", 0.01),
        max_position_pct=_env("MAX_POSITION_PCT", 0.20),
        max_open_positions=_env("MAX_OPEN_POSITIONS", 5),
        max_drawdown_pct=_env("MAX_DRAWDOWN_PCT", 0.20),
        max_daily_loss_pct=_env("MAX_DAILY_LOSS_PCT", 0.03),
        kelly_fraction=_env("KELLY_FRACTION", 0.25),
        max_symbol_weight=_env("MAX_SYMBOL_WEIGHT", 0.35),
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no attribute {key!r}")
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg
