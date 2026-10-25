"""FastAPI service exposing the bot over HTTP.

Every endpoint is read-only with respect to money. There is no endpoint that
places a live order; the live broker cannot even be constructed without the
environment flag and the confirmation phrase, so the HTTP surface cannot reach
real funds by design rather than by convention.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from .backtest.walkforward import compare_strategies, walk_forward
from .broker.live import describe_live_status
from .config import load_config
from .data import Universe
from .strategies import available, describe

# FastAPI is imported lazily on purpose. The CLI, the TUI, the backtester, the
# paper broker, and every test import this package; none of them need FastAPI,
# and a module-level import meant that a single missing dependency made
# `marketbot.api` — and anything importing it — fail with an opaque error.
_FASTAPI_IMPORT_ERROR: Optional[Exception] = None
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover - depends on the environment
    _FASTAPI_IMPORT_ERROR = e


def fastapi_status() -> dict:
    """Report whether the HTTP API can run, and why not if it cannot.

    The message names the interpreter to check, because "fastapi is missing"
    almost always means the wrong Python is being used rather than a broken
    install.
    """
    import sys

    if _FASTAPI_IMPORT_ERROR is None:
        import fastapi

        return {
            "available": True,
            "version": getattr(fastapi, "__version__", "unknown"),
            "python": sys.executable,
        }
    return {
        "available": False,
        "python": sys.executable,
        "error": str(_FASTAPI_IMPORT_ERROR),
        "diagnostic": (
            'Run: python -c "import sys, fastapi; print(sys.executable, fastapi.__version__)"'
        ),
        "note": (
            "The CLI, TUI, backtester and paper broker do not need FastAPI."
        ),
    }


def _require_fastapi():
    if _FASTAPI_IMPORT_ERROR is not None:
        raise ImportError(
            "FastAPI is required for the HTTP API but could not be imported:\n"
            f"  {_FASTAPI_IMPORT_ERROR}\n\n"
            f"Interpreter in use: {fastapi_status()['python']}\n"
            f"{fastapi_status()['diagnostic']}\n\n"
            "Everything else runs without it:\n"
            "  python -m marketbot backtest --symbol BTC-USD --strategy sma_crossover\n"
            "  python -m marketbot.tui_main\n"
            "  ./verify.sh"
        ) from _FASTAPI_IMPORT_ERROR


@lru_cache(maxsize=4)
def _config(**overrides):
    return load_config(**overrides)


def create_app(config=None):
    """Build the FastAPI app. Raises ImportError with a diagnosis if FastAPI is absent."""
    _require_fastapi()
    app = FastAPI(
        title="MarketBot",
        version="0.1.0",
        description="Backtest, paper trade, and (explicitly) live trade crypto and stocks.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = {"config": config or _config()}

    # ------------------------------------------------------------------
    # static dashboard
    # ------------------------------------------------------------------
    dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

    @app.get("/")
    def index():
        index_file = dashboard_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"service": "MarketBot", "docs": "/docs"}

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"status": "ok", "config": {
            "starting_cash": state["config"].starting_cash,
            "fee_rate": state["config"].fee_rate,
            "slippage_rate": state["config"].slippage_rate,
            "risk_per_trade": state["config"].risk_per_trade,
            "live_trading": state["config"].live_trading,
        }}

    @app.get("/api/live-status")
    def live_status():
        return describe_live_status()

    @app.get("/api/strategies")
    def strategies():
        return {
            "count": len(available()),
            "strategies": [
                describe(name) for name in available()
            ],
        }

    @app.get("/api/indicators")
    def indicators():
        from .indicators import available as inds

        return {"count": len(inds()), "indicators": inds()}

    @app.get("/api/universe")
    def universe():
        with Universe(state["config"]) as u:
            assets = u.load() or u.defaults(include_stocks=True)
        return {"count": len(assets), "assets": [a.symbol for a in assets]}

    # ------------------------------------------------------------------
    # market data
    # ------------------------------------------------------------------
    def _candles(symbol: str, interval: str, bars: int):
        with Universe(state["config"]) as u:
            return u.yahoo.fetch_candles(symbol, interval=interval, limit=bars)

    @app.get("/api/market/{symbol}")
    def market(symbol: str, interval: str = "1d", bars: int = 365):
        try:
            candles = _candles(symbol, interval, bars)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {
            "symbol": symbol,
            "interval": interval,
            "bars": len(candles),
            "first_ts": candles[0].ts,
            "last_ts": candles[-1].ts,
            "last_close": candles[-1].close,
            "candles": [c.to_dict() for c in candles],
        }

    @app.get("/api/indicators/{symbol}")
    def indicator_values(symbol: str, name: str, period: int = 14, bars: int = 365):
        from .indicators import compute

        try:
            candles = _candles(symbol, "1d", bars)
            series = compute(name, candles, period)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        defined = [v for v in series if v is not None]
        return {
            "symbol": symbol,
            "indicator": name,
            "period": period,
            "points": len(series),
            "defined": len(defined),
            "last": round(series[-1], 6) if series[-1] is not None else None,
            "series": [round(v, 6) if v is not None else None for v in series],
            "timestamps": [c.ts for c in candles],
        }

    # ------------------------------------------------------------------
    # backtests
    # ------------------------------------------------------------------
    class BacktestRequest(BaseModel):
        symbol: str = "BTC-USD"
        strategy: str = "sma_crossover"
        params: dict = {}
        interval: str = "1d"
        bars: int = 400
        starting_cash: Optional[float] = None

        class Config:
            extra = "forbid"

    @app.post("/api/backtest")
    def backtest(request: BacktestRequest):
        from .backtest import run_backtest
        from .strategies import get

        try:
            strategy = get(request.strategy, **request.params)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        overrides = {}
        if request.starting_cash is not None:
            overrides["starting_cash"] = request.starting_cash
        cfg = load_config(**overrides) if overrides else state["config"]
        try:
            candles = _candles(request.symbol, request.interval, request.bars)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        result = run_backtest(request.symbol, candles, strategy, cfg)
        return result.to_dict()

    @app.get("/api/compare")
    def compare(symbol: str = "BTC-USD", interval: str = "1d", bars: int = 400,
               strategies: Optional[str] = Query(default=None)):
        names = strategies.split(",") if strategies else available()
        try:
            candles = _candles(symbol, interval, bars)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"symbol": symbol, "rows": compare_strategies(symbol, candles, names, state["config"])}

    class OptimizeRequest(BaseModel):
        symbol: str = "BTC-USD"
        strategy: str = "sma_crossover"
        grid: dict = {"fast": [5, 10, 20], "slow": [30, 50, 100]}
        train_bars: int = 250
        test_bars: int = 60
        interval: str = "1d"
        bars: int = 1200

    @app.post("/api/optimize")
    def optimize(request: OptimizeRequest):
        from .backtest.walkforward import parameter_stability

        try:
            candles = _candles(request.symbol, request.interval, request.bars)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        wf = walk_forward(request.symbol, candles, request.strategy, request.grid,
                          state["config"], train_bars=request.train_bars,
                          test_bars=request.test_bars)
        return {**wf.to_dict(), "stability": parameter_stability(wf)}

    # ------------------------------------------------------------------
    # AI
    # ------------------------------------------------------------------
    class FirmRequest(BaseModel):
        symbol: str = "BTC-USD"
        interval: str = "1d"
        bars: int = 400
        kind: str = "crypto"
        advisor: str = "local"

    @app.post("/api/firm")
    def firm(request: FirmRequest):
        from .ai import AnalysisRequest, make_firm

        try:
            candles = _candles(request.symbol, request.interval, request.bars)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        req = AnalysisRequest(
            symbol=request.symbol,
            candles=candles,
            strategy_options=available(),
            constraints={"kind": request.kind},
        )
        f = make_firm(state["config"], advisor_name=request.advisor)
        return f.run(req)

    class AdviseRequest(BaseModel):
        symbol: str = "BTC-USD"
        interval: str = "1d"
        bars: int = 400
        advisor: str = "local"
        question: str = "What should I trade?"

    @app.post("/api/advise")
    def advise(request: AdviseRequest):
        from .ai import AnalysisRequest, make_advisor

        try:
            candles = _candles(request.symbol, request.interval, request.bars)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        req = AnalysisRequest(symbol=request.symbol, candles=candles,
                              strategy_options=available(), question=request.question)
        advisor = make_advisor(request.advisor)
        return advisor.advise(req).to_dict()

    return app


def get_app():
    """Return a shared app instance, building it on first use.

    The app is built lazily rather than at import time so that importing
    `marketbot.api` never requires FastAPI. `python -m marketbot.api` calls this
    through `main()`, and a bare `import marketbot.api` stays side-effect free.
    """
    global app
    if app is None:
        app = create_app()
    return app


app = None


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(get_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
