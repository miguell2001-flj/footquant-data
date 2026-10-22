"""Phase 5 - fractional Kelly staking."""

from .kelly import StakeRecommendation, allocate_stakes, expected_log_growth, kelly_fraction, kelly_stake

__all__ = ["StakeRecommendation", "allocate_stakes", "expected_log_growth", "kelly_fraction", "kelly_stake"]
