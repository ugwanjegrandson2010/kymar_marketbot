"""Data layer tests: providers, cache, and universe.

Network tests are skipped when the API is unreachable, so the suite passes
offline; everything that can be checked without a socket is checked regardless.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketbot.config import load_config
from marketbot.data.base import DataError, Http
from marketbot.data.cache import CandleCache
from marketbot.model import Candle

NETWORK_AVAILABLE = None


def network_available() -> bool:
    """Probe once, cache the result, and never let a network failure fail a test."""
    global NETWORK_AVAILABLE
    if NETWORK_AVAILABLE is None:
        try:
            import urllib.request

            req = urllib.request.Request(
                "https://api.coingecko.com/api/v3/ping",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                NETWORK_AVAILABLE = r.status == 200
        except Exception:  # noqa: BLE001
            NETWORK_AVAILABLE = False
    return NETWORK_AVAILABLE


def requires_network(test):
    return unittest.skipUnless(network_available(), "network/API unavailable")(test)


class TestHttp(unittest.TestCase):
    def test_404_raises_data_error(self):
        http = Http(load_config())
        with self.assertRaises(DataError):
            http.get_json("https://api.coingecko.com/api/v3/this_endpoint_does_not_exist")

    def test_bad_host_raises_data_error(self):
        http = Http(load_config(max_retries=1, retry_backoff=0.01))
        with self.assertRaises(DataError):
            http.get_json("http://127.0.0.1:9/nothing")

    @requires_network
    def test_sends_coingecko_demo_key_header_when_configured(self):
        cfg = load_config(coingecko_api_key="test-key-value")
        http = Http(cfg)
        self.assertEqual(http.extra_headers.get("x-cg-demo-api-key"), "test-key-value")

    def test_no_header_without_a_key(self):
        http = Http(load_config(coingecko_api_key=""))
        self.assertNotIn("x-cg-demo-api-key", http.extra_headers)


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "cache.db"
        self.cfg = load_config(cache_db=self.path, cache_ttl_seconds=60)
        self.cache = CandleCache(self.cfg)
        self.candles = [
            Candle(ts=1_600_000_000 + i * 86400, open=100 + i, high=101 + i,
                   low=99 + i, close=100 + i, volume=1000)
            for i in range(50)
        ]

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_round_trip_preserves_every_bar(self):
        written = self.cache.put("test", "ABC", "1d", self.candles)
        self.assertEqual(written, 50)
        read = self.cache.get("test", "ABC", "1d")
        self.assertEqual(len(read), 50)
        self.assertEqual(read[0].close, self.candles[0].close)
        self.assertEqual(read[-1].ts, self.candles[-1].ts)

    def test_put_deduplicates_on_timestamp(self):
        self.cache.put("test", "ABC", "1d", self.candles)
        self.cache.put("test", "ABC", "1d", self.candles)
        self.assertEqual(len(self.cache.get("test", "ABC", "1d")), 50)

    def test_time_range_filtering(self):
        self.cache.put("test", "ABC", "1d", self.candles)
        lo = self.candles[10].ts
        hi = self.candles[20].ts
        read = self.cache.get("test", "ABC", "1d", min_ts=lo, max_ts=hi)
        self.assertEqual(len(read), 11)

    def test_is_fresh_reflects_ttl(self):
        self.cache.put("test", "ABC", "1d", self.candles)
        self.assertTrue(self.cache.is_fresh("test", "ABC", "1d"))
        stale = CandleCache(load_config(cache_db=self.path, cache_ttl_seconds=-1))
        self.assertFalse(stale.is_fresh("test", "ABC", "1d"))
        stale.close()

    def test_coverage_reports_bounds(self):
        self.cache.put("test", "ABC", "1d", self.candles)
        cov = self.cache.coverage("test", "ABC", "1d")
        self.assertEqual(cov["bars"], 50)
        self.assertEqual(cov["first_ts"], self.candles[0].ts)
        self.assertEqual(cov["last_ts"], self.candles[-1].ts)

    def test_invalidate_removes_rows(self):
        self.cache.put("test", "ABC", "1d", self.candles)
        removed = self.cache.invalidate("test", "ABC", "1d")
        self.assertEqual(removed, 50)
        self.assertEqual(self.cache.get("test", "ABC", "1d"), [])

    def test_corrupt_rows_are_dropped_on_read(self):
        # A bar with a non-positive close must never reach a strategy.
        bad = [Candle(ts=1, open=-5, high=-4, low=-6, close=-5, volume=0)]
        self.cache.put("test", "BAD", "1d", bad)
        self.assertEqual(self.cache.get("test", "BAD", "1d"), [])

    def test_stats_counts_rows_and_symbols(self):
        self.cache.put("p1", "A", "1d", self.candles[:10])
        self.cache.put("p1", "B", "1d", self.candles[:20])
        stats = self.cache.stats()
        self.assertEqual(stats["bars"], 30)
        self.assertEqual(stats["symbols"], 2)


class TestProviders(unittest.TestCase):
    @requires_network
    def test_coingecko_ping(self):
        from marketbot.data import CoinGeckoProvider

        provider = CoinGeckoProvider(load_config())
        health = provider.health()
        self.assertTrue(health["ok"], health)

    @requires_network
    def test_coingecko_fetch_returns_aligned_candles(self):
        from marketbot.data import CoinGeckoProvider

        provider = CoinGeckoProvider(load_config())
        candles = provider.fetch_candles("bitcoin", interval="1d", limit=60)
        self.assertEqual(len(candles), 60)
        for i in range(1, len(candles)):
            self.assertGreater(candles[i].ts, candles[i - 1].ts)

    @requires_network
    def test_coingecko_top_coins_are_ranked_and_liquid(self):
        from marketbot.data import CoinGeckoProvider

        provider = CoinGeckoProvider(load_config())
        coins = provider.top_coins(per_page=10, pages=1)
        self.assertEqual(len(coins), 10)
        self.assertTrue(all(c["volume"] > 0 for c in coins))

    @requires_network
    def test_coingecko_unknown_coin_raises(self):
        from marketbot.data import CoinGeckoProvider

        provider = CoinGeckoProvider(load_config())
        with self.assertRaises(DataError):
            provider.fetch_candles("this_coin_does_not_exist_xyz", limit=10)

    @requires_network
    def test_yahoo_fetch_returns_ohlcv(self):
        from marketbot.data import YahooProvider

        provider = YahooProvider(load_config())
        candles = provider.fetch_candles("BTC-USD", interval="1d", limit=100)
        self.assertEqual(len(candles), 100)
        # Yahoo supplies a real high/low range; CoinGecko's market_chart does not.
        self.assertTrue(any(c.high > c.low for c in candles))

    @requires_network
    def test_yahoo_resolves_tickers(self):
        from marketbot.data import YahooProvider

        provider = YahooProvider(load_config())
        self.assertEqual(provider.resolve("BTC"), "BTC-USD")
        self.assertEqual(provider.resolve("AAPL"), "AAPL")
        self.assertEqual(provider.resolve("BTC-USD"), "BTC-USD")

    @requires_network
    def test_yahoo_stock_data(self):
        from marketbot.data import YahooProvider

        candles = YahooProvider(load_config()).fetch_candles("AAPL", interval="1d", limit=50)
        self.assertEqual(len(candles), 50)
        self.assertTrue(all(c.close > 0 for c in candles))


class TestUniverse(unittest.TestCase):
    def test_defaults_include_both_asset_classes(self):
        from marketbot.data import Universe

        assets = Universe().defaults(include_stocks=True)
        kinds = {a.kind for a in assets}
        self.assertEqual(kinds, {"crypto", "stock"})

    def test_crypto_only_excludes_stocks(self):
        from marketbot.data import Universe

        assets = Universe().defaults(include_stocks=False)
        self.assertTrue(all(a.kind == "crypto" for a in assets))

    def test_save_and_load_round_trip(self):
        from marketbot.data import Universe

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(cache_db=Path(tmp) / "cache.db")
            u = Universe(cfg)
            assets = u.defaults(include_stocks=False)[:5]
            path = u.save(assets, Path(tmp) / "universe.json")
            loaded = u.load(path)
            self.assertEqual([a.symbol for a in loaded], [a.symbol for a in assets])
            u.close()


class TestConfig(unittest.TestCase):
    def test_live_trading_defaults_off(self):
        cfg = load_config()
        self.assertFalse(cfg.live_trading)

    def test_overrides_apply(self):
        cfg = load_config(starting_cash=1234.0, fee_rate=0.02)
        self.assertEqual(cfg.starting_cash, 1234.0)
        self.assertEqual(cfg.fee_rate, 0.02)

    def test_unknown_override_raises(self):
        with self.assertRaises(AttributeError):
            load_config(not_a_real_setting=1)

    def test_env_overrides_values(self):
        import os

        os.environ["MARKETBOT_STARTING_CASH"] = "7777"
        try:
            cfg = load_config()
            self.assertEqual(cfg.starting_cash, 7777.0)
        finally:
            del os.environ["MARKETBOT_STARTING_CASH"]

    def test_env_file_in_project_root_is_loaded(self):
        """A .env in the project root must be read even when CWD is elsewhere.

        This was a real bug: config.py read only os.environ, so a key placed in
        .env was silently ignored.
        """
        import pathlib
        import subprocess
        import sys
        import tempfile

        from marketbot import config as cfgmod

        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, ".env").write_text(
                "MARKETBOT_STARTING_CASH=31337\n"
                "MARKETBOT_FEE_RATE=0.03\n"
            )
            code = (
                "import sys, os; sys.path.insert(0, %r); os.chdir(%r)\n"
                "from marketbot.config import load_config\n"
                "c = load_config()\n"
                "print(c.starting_cash, c.fee_rate)\n"
            ) % (str(cfgmod.PROJECT_ROOT.parent), tmp)
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(out.stdout.strip(), "31337.0 0.03")

    def test_real_environment_beats_env_file(self):
        """An explicit export must override the file, not the reverse."""
        import pathlib
        import subprocess
        import sys
        import tempfile

        from marketbot import config as cfgmod

        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, ".env").write_text("MARKETBOT_STARTING_CASH=1111\n")
            code = (
                "import sys, os; sys.path.insert(0, %r); os.chdir(%r)\n"
                "os.environ['MARKETBOT_STARTING_CASH'] = '2222'\n"
                "from marketbot.config import load_config\n"
                "print(load_config().starting_cash)\n"
            ) % (str(cfgmod.PROJECT_ROOT.parent), tmp)
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(out.stdout.strip(), "2222.0")

    def test_repr_never_leaks_a_credential(self):
        """A stray repr or log line must not expose the key."""
        cfg = load_config(coingecko_api_key="CG-LEAKTEST1234567890")
        self.assertNotIn("CG-LEAKTEST1234567890", repr(cfg))
        self.assertNotIn("CG-LEAKTEST1234567890", str(cfg.masked_summary()))
        self.assertTrue(cfg.masked_summary()["has_coingecko_key"])

    def test_env_file_is_gitignored(self):
        """The file that holds credentials must never be tracked."""
        import subprocess

        root = pathlib.Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "-C", str(root), "check-ignore", "-v", ".env"],
                             capture_output=True, text=True)
        self.assertIn(".env", out.stdout)


if __name__ == "__main__":
    unittest.main()
