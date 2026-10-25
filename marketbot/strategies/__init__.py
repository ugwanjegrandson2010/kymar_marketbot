"""Strategy library: registry and lookup by name.

Strategies are registered by class so the CLI and API can select one by string
without importing every module, and so an unknown name fails loudly instead of
silently doing nothing.
"""

from __future__ import annotations

from .base import Strategy, validate_params
from .legendary import (
    BulkowskiPatterns,
    Connors2Rsi,
    DruckenmillerMomentum,
    ElderTripleScreen,
    LivermorePivot,
    MinerviniTemplate,
    PringKST,
    Raschke8020,
    TudorJones200Sma,
    TurtleBreakout,
    WilderOriginal,
)
from .meta import CompositeStrategy, WeightedComposite
from .momentum import (
    CciStrategy,
    RsiAtrBreakout,
    RsiMeanReversion,
    RsiTrendContinuation,
    StochasticStrategy,
    WilliamsRStrategy,
)
from .trend import (
    AdxTrend,
    DonchianBreakout,
    EmaTrend,
    IchimokuStrategy,
    MacdStrategy,
    SmaCrossover,
    SupertrendStrategy,
)
from .volatility import (
    AtrVolatilityBreakout,
    BollingerBreakout,
    BollingerReversion,
    KeltnerBreakout,
    SqueezeBreakout,
    StdevRankStrategy,
)
from .volume import MfiStrategy, ObvDivergence, VolumePriceTrend, VwapMeanReversion

STRATEGIES: dict[str, type[Strategy]] = {
    # trend
    SmaCrossover.name: SmaCrossover,
    EmaTrend.name: EmaTrend,
    MacdStrategy.name: MacdStrategy,
    AdxTrend.name: AdxTrend,
    SupertrendStrategy.name: SupertrendStrategy,
    DonchianBreakout.name: DonchianBreakout,
    IchimokuStrategy.name: IchimokuStrategy,
    # momentum
    RsiMeanReversion.name: RsiMeanReversion,
    RsiTrendContinuation.name: RsiTrendContinuation,
    StochasticStrategy.name: StochasticStrategy,
    WilliamsRStrategy.name: WilliamsRStrategy,
    CciStrategy.name: CciStrategy,
    RsiAtrBreakout.name: RsiAtrBreakout,
    # volatility
    BollingerBreakout.name: BollingerBreakout,
    BollingerReversion.name: BollingerReversion,
    SqueezeBreakout.name: SqueezeBreakout,
    KeltnerBreakout.name: KeltnerBreakout,
    StdevRankStrategy.name: StdevRankStrategy,
    AtrVolatilityBreakout.name: AtrVolatilityBreakout,
    # volume
    VwapMeanReversion.name: VwapMeanReversion,
    MfiStrategy.name: MfiStrategy,
    ObvDivergence.name: ObvDivergence,
    VolumePriceTrend.name: VolumePriceTrend,
    # attributed to specific traders
    TurtleBreakout.name: TurtleBreakout,
    TudorJones200Sma.name: TudorJones200Sma,
    Raschke8020.name: Raschke8020,
    ElderTripleScreen.name: ElderTripleScreen,
    WilderOriginal.name: WilderOriginal,
    PringKST.name: PringKST,
    MinerviniTemplate.name: MinerviniTemplate,
    Connors2Rsi.name: Connors2Rsi,
    DruckenmillerMomentum.name: DruckenmillerMomentum,
    LivermorePivot.name: LivermorePivot,
    BulkowskiPatterns.name: BulkowskiPatterns,
    # composite
    CompositeStrategy.name: CompositeStrategy,
    WeightedComposite.name: WeightedComposite,
}


def available() -> list[str]:
    return sorted(STRATEGIES.keys())


def get(name: str, **params) -> Strategy:
    """Instantiate a strategy by name.

    Unknown names raise KeyError; a mistyped parameter raises ValueError from
    validate_params. Both fail at call time rather than degrading to defaults,
    because a silent parameter typo invalidates the whole backtest.
    """
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    cls = STRATEGIES[name]
    merged = validate_params(cls, params)
    return cls(**merged)


def describe(name: str) -> dict:
    cls = STRATEGIES.get(name)
    if cls is None:
        raise KeyError(f"unknown strategy {name!r}")
    return {
        "name": cls.name,
        "description": cls.description,
        "params": cls.default_params(),
        "indicators": list(cls.indicators),
    }


__all__ = ["STRATEGIES", "Strategy", "available", "describe", "get", "validate_params"]
