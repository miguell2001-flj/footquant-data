"""Dispatcher for the early-payout path model chosen in ValueConfig."""

from __future__ import annotations

from ..config import ValueConfig
from .score_effects import ScoreEffectModel, default_score_effects
from .score_matrix import early_payout_probs


def simulate_early_payout(lam: float, mu: float, cfg: ValueConfig, seed: int | None = None, n_sims: int | None = None,
                          score_effects: ScoreEffectModel | None = None) -> dict[str, float]:
    """P(reached a 2-goal lead), P(reached and did not win) for both sides under
    the configured path model: 'poisson' (constant rates) or 'score_effects'
    (time- and score-dependent rates, the package default model)."""
    n = n_sims or cfg.n_sims
    if cfg.early_payout_model == "poisson":
        return early_payout_probs(lam, mu, cfg.early_payout_lead, n, seed=seed)
    if score_effects is None:
        try:
            score_effects = default_score_effects()
        except FileNotFoundError:
            import warnings

            warnings.warn("score-effects model file missing; falling back to the constant-rate path model")
            return early_payout_probs(lam, mu, cfg.early_payout_lead, n, seed=seed)
    return score_effects.simulate(lam, mu, cfg.early_payout_lead, n, seed=seed)
