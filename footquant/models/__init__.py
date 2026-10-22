"""Phases 2 & 3 - Dixon-Coles engine, score matrix, boosted rate adjuster."""

from .context_boost import ContextualRateAdjuster, team_rows_from_predictions
from .dixon_coles import DCParams, DixonColesModel, tau, time_weights
from .score_matrix import early_payout_probs, market_probs, score_matrix

__all__ = [
    "ContextualRateAdjuster",
    "DCParams",
    "DixonColesModel",
    "early_payout_probs",
    "market_probs",
    "score_matrix",
    "tau",
    "team_rows_from_predictions",
    "time_weights",
]
