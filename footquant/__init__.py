"""footquant - xG-driven Dixon-Coles engine, contextual boosting, bet365 value detection
and fractional-Kelly staking for European football markets.

Package layout (mirrors the six-phase framework):

    data/      Phase 1 - schemas, ingestion, xG aggregation, Elo, contextual features
    models/    Phase 2 & 3 - Dixon-Coles MLE engine, score matrix, boosted rate adjuster
    market/    Phase 4 - de-vigging, odds feeds, EV / early-payout value logic
    staking/   Phase 5 - fractional Kelly allocation
    backtest/  Phase 6 - Brier / log-loss / RPS, walk-forward engine, xi tuning, P&L
    pipeline   End-to-end orchestration (fit -> predict -> value -> stake)
    simulate   Synthetic league generator used by the tests and the demo
"""

from .config import BacktestConfig, EloConfig, ModelConfig, StakingConfig, ValueConfig
from .paper_trading import Ledger, fit_pool_weight_on_history, make_bet_sheet
from .pipeline import BettingPipeline

__version__ = "0.6.0"

__all__ = [
    "BacktestConfig",
    "BettingPipeline",
    "EloConfig",
    "Ledger",
    "ModelConfig",
    "StakingConfig",
    "ValueConfig",
    "fit_pool_weight_on_history",
    "make_bet_sheet",
]
