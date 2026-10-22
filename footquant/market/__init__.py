"""Phase 4 - de-vigging, odds feeds, value detection."""

from .blend import MultinomialStacker, fit_pool_weight, log_linear_pool, market_implied_rates
from .devig import devig, implied_probabilities, overround
from .odds_feed import OddsFeed, OddsSnapshot, snapshots_from_frame
from .value import ValueBet, evaluate_snapshot, expected_value, value_table

__all__ = [
    "MultinomialStacker",
    "fit_pool_weight",
    "log_linear_pool",
    "market_implied_rates",
    "OddsFeed",
    "OddsSnapshot",
    "ValueBet",
    "devig",
    "evaluate_snapshot",
    "expected_value",
    "implied_probabilities",
    "overround",
    "snapshots_from_frame",
    "value_table",
]
