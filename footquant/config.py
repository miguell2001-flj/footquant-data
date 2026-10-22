"""Configuration objects for every stage of the system.

All configs are frozen dataclasses so that a backtest run is fully described by
its config values (reproducibility) and nothing mutates them mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass

# Phase 5: the only two multipliers the staking module accepts. Full Kelly is
# deliberately not an option - model error makes it over-bet in practice.
KELLY_FRACTIONS: dict[str, float] = {"half": 0.5, "quarter": 0.25}

# Competitions where bet365's "2 Goals Ahead Early Payout" applies to pre-match
# Full Time Result bets (home or away win; draws excluded). The identifiers are
# the `league` codes used in the match table - map your own codes onto these.
DEFAULT_EARLY_PAYOUT_COMPETITIONS: frozenset[str] = frozenset(
    {
        "ENG-Premier League",
        "ENG-Championship",
        "ESP-La Liga",
        "GER-Bundesliga",
        "ITA-Serie A",
        "FRA-Ligue 1",
        "UEFA-Champions League",
        "UEFA-Europa League",
        "UEFA-Conference League",
    }
)


@dataclass(frozen=True)
class ModelConfig:
    """Dixon-Coles engine settings (Phase 2)."""

    xi: float = 0.0018
    """Time-decay rate per day. 0.0018/day is the per-day equivalent of the
    0.0065 per half-week reported by Dixon & Coles (1997); half-life ~ 385 days.
    Re-tune per league with `backtest.walkforward.tune_xi` (RPS criterion)."""

    xg_weight: float = 0.6
    """Weight of xG in the fit target: target = w * xG + (1 - w) * goals.
    0.0 reproduces the classical goals-only Dixon-Coles model, 1.0 fits rates
    to xG only. The tau dependence term always uses observed goals."""

    max_goals: int = 8
    """Score matrix covers 0..max_goals goals per team (8 -> the 9x9 grid)."""

    rho_bounds: tuple[float, float] = (-0.3, 0.2)
    """Box bounds for the Dixon-Coles dependence parameter rho."""

    ridge: float = 0.01
    """L2 penalty on attack/defence parameters. Small values stabilise early-season
    fits and teams with few matches in the window."""

    unknown_team_prior: tuple[float, float] = (-0.10, 0.10)
    """(attack, defence) used for a team absent from the fitted window, e.g. a
    promoted side before its first match. Slightly below-average by default."""

    max_iter: int = 500

    def __post_init__(self) -> None:
        if not 0.0 <= self.xg_weight <= 1.0:
            raise ValueError("xg_weight must be in [0, 1]")
        if self.xi < 0:
            raise ValueError("xi must be >= 0")
        if self.max_goals < 5:
            raise ValueError("max_goals must be >= 5")
        if self.rho_bounds[0] >= self.rho_bounds[1]:
            raise ValueError("rho_bounds must be (low, high)")


@dataclass(frozen=True)
class EloConfig:
    """League-separated Elo settings (Phase 1)."""

    k_factor: float = 20.0
    home_adv: float = 65.0
    """Home advantage in Elo points added to the home rating before computing
    the expected score (~ +0.09 expected points at equal ratings)."""
    init_rating: float = 1500.0
    mov_multiplier: bool = True
    """Scale K by margin of victory: 1 for a 1-goal margin, 1.5 for 2 goals,
    (11 + N) / 8 for N >= 3 (the eloratings.net convention)."""
    result_from: str = "goals"
    """'goals' -> classical 1 / 0.5 / 0 result; 'xg' -> expected points share
    P(X > Y) + 0.5 P(X = Y) with X ~ Pois(home xG), Y ~ Pois(away xG)."""

    def __post_init__(self) -> None:
        if self.result_from not in {"goals", "xg"}:
            raise ValueError("result_from must be 'goals' or 'xg'")


@dataclass(frozen=True)
class ValueConfig:
    """Value detection settings (Phase 4)."""

    edge_margin: float = 0.03
    """Flag a bet only if p_model - p_fair >= edge_margin (the 3% margin of error)."""
    market_shrink: float = 0.0
    """Weight on the de-vigged market price in the probability used for edge
    and staking: p_used = (1 - s) * p_model + s * p_fair. 0 = pure model (the
    framework's specification). Values of 0.3-0.6 are the standard defence
    against a model's own estimation noise masquerading as edge, and reduce
    Kelly stakes accordingly."""
    devig_method: str = "shin"
    """'shin' | 'power' | 'multiplicative'. Shin's method corrects the
    favourite-longshot bias that proportional normalisation ignores."""
    markets: frozenset[str] = frozenset({"1X2", "BTTS", "OU25"})
    """Markets the value engine prices. On EPL 2019-21 the Dixon-Coles totals
    forecast lost 19% of turnover against bet365's O/U 2.5 line (t = -3.2), so
    consider frozenset({"1X2"}) until the totals model is improved."""
    early_payout: bool = True
    early_payout_lead: int = 2
    early_payout_model: str = "score_effects"
    """'score_effects' (time- and score-dependent goal intensities fitted on the
    big-5 leagues, the default) or 'poisson' (constant rates)."""
    early_payout_scale: float = 1.5
    """Multiplier on the simulated increment for a side the market rates at 50%
    or less. The path model under-predicts "led by two, did not win" because
    trailing teams push and leaders sit back: on the big-5 goal timelines of
    2025-26 and 2026-27 (3,900 team-matches, ESPN) the realised rate was
    2.4% +- 0.25 against 1.5% simulated for sides priced under 60%, and
    0.9% against 1.1% for sides priced at 70% or more. The scale falls
    linearly from `early_payout_scale` at 50% to `early_payout_scale_favourite`
    at 70% (see `early_payout_multiplier`). 1.0 / 1.0 restores the raw simulation."""
    early_payout_scale_favourite: float = 0.9
    """Multiplier for a side the market rates at 70% or more (see above)."""
    edge_metric: str = "ev"
    """What `edge_margin` is measured on when a bet sheet is made:
    'ev'     - the expected return per unit staked at the price on offer,
               p_paid * odds - 1 (the value after the bookmaker's margin);
    'points' - p_paid - p_fair, the paid-out probability's lead over the
               de-vigged market in probability points (ignores the price, so
               with a pooled probability close to the market it is met by
               almost any early-payout increment)."""

    def early_payout_multiplier(self, p_market: float) -> float:
        """Calibration multiplier on the simulated early-payout increment for a side the market rates at `p_market`."""
        lo, hi = 0.5, 0.7
        if p_market <= lo:
            return self.early_payout_scale
        if p_market >= hi:
            return self.early_payout_scale_favourite
        t = (p_market - lo) / (hi - lo)
        return self.early_payout_scale + t * (self.early_payout_scale_favourite - self.early_payout_scale)
    early_payout_competitions: frozenset[str] = DEFAULT_EARLY_PAYOUT_COMPETITIONS
    n_sims: int = 200_000
    """Monte-Carlo paths used to estimate the probability a team leads by
    `early_payout_lead` at some point during the match."""
    seed: int = 7

    def __post_init__(self) -> None:
        if self.devig_method not in {"shin", "power", "multiplicative"}:
            raise ValueError("devig_method must be 'shin', 'power' or 'multiplicative'")
        if self.edge_margin < 0:
            raise ValueError("edge_margin must be >= 0")
        if not 0.0 <= self.market_shrink < 1.0:
            raise ValueError("market_shrink must be in [0, 1)")
        if self.early_payout_model not in {"poisson", "score_effects"}:
            raise ValueError("early_payout_model must be 'poisson' or 'score_effects'")
        if self.edge_metric not in {"ev", "points"}:
            raise ValueError("edge_metric must be 'ev' or 'points'")


@dataclass(frozen=True)
class StakingConfig:
    """Fractional Kelly settings (Phase 5)."""

    kelly_mode: str = "quarter"
    """'half' or 'quarter' - the only accepted multipliers (see KELLY_FRACTIONS)."""
    max_stake_frac: float = 0.05
    """Hard cap on any single stake as a fraction of bankroll."""
    max_total_frac: float = 0.20
    """Cap on total open exposure across simultaneous bets; stakes are scaled
    down proportionally if the sum of fractional-Kelly stakes exceeds it."""
    min_stake: float = 0.0
    round_to: float = 0.5
    """Stakes are rounded down to this currency increment."""

    def __post_init__(self) -> None:
        if self.kelly_mode not in KELLY_FRACTIONS:
            raise ValueError(f"kelly_mode must be one of {sorted(KELLY_FRACTIONS)}")
        if not 0 < self.max_stake_frac <= 1:
            raise ValueError("max_stake_frac must be in (0, 1]")

    @property
    def kelly_fraction(self) -> float:
        return KELLY_FRACTIONS[self.kelly_mode]


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward backtest settings (Phase 6)."""

    refit_every_days: int = 7
    """Re-estimate the model at the start of every window of this length."""
    min_train_matches: int = 150
    """Do not start predicting a league until this many matches are available."""
    xi_grid: tuple[float, ...] = (0.0, 0.001, 0.0018, 0.003, 0.005)
    bankroll: float = 1000.0
