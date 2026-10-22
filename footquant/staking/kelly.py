"""Kelly criterion staking (Phase 5).

For a binary bet at decimal odds o with model win probability p:

    b  = o - 1                       net odds
    q  = 1 - p
    f* = (b p - q) / b               Kelly (1956) optimal fraction of bankroll
    g(f) = p ln(1 + b f) + q ln(1 - f)   expected log growth per bet

f* > 0 iff p o > 1, i.e. iff EV > 0. The module stakes `fraction * f*` with
`fraction` hard-wired to 0.5 (half) or 0.25 (quarter) - see config.KELLY_FRACTIONS -
because f* is computed from an estimated p, and estimation error makes full
Kelly over-bet: at fraction c the long-run growth rate is c(2 - c) of the full
Kelly growth (so quarter Kelly keeps ~44% of the growth at 1/16 the variance of
log wealth, half Kelly 75% of the growth at 1/4 the variance).

Simultaneous bets: exact multi-bet Kelly needs a joint optimisation over all
outcome combinations. For small independent stakes, per-bet fractional Kelly
with a cap on total exposure is the standard approximation used here; at most
one selection per (match, market) is staked (the highest EV), since the others
are mutually exclusive with it.

For early-payout bets `p` is the probability of being *paid*, which is the
quantity the bet's cash-flow actually depends on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import StakingConfig
from ..market.value import ValueBet


def kelly_fraction(p: float, odds: float) -> float:
    """Full-Kelly fraction f* = (b p - q) / b, floored at 0."""
    if odds <= 1.0:
        raise ValueError("decimal odds must be > 1")
    b = odds - 1.0
    q = 1.0 - p
    return max((b * p - q) / b, 0.0)


def expected_log_growth(f: float, p: float, odds: float) -> float:
    b = odds - 1.0
    if f >= 1.0 or f < 0.0:
        return -np.inf
    return p * np.log1p(b * f) + (1.0 - p) * np.log1p(-f)


def kelly_stake(p: float, odds: float, bankroll: float, cfg: StakingConfig | None = None) -> float:
    """Fractional-Kelly stake in currency, capped and rounded down."""
    cfg = cfg or StakingConfig()
    f = kelly_fraction(p, odds) * cfg.kelly_fraction
    f = min(f, cfg.max_stake_frac)
    stake = bankroll * f
    if cfg.round_to > 0:
        stake = np.floor(stake / cfg.round_to) * cfg.round_to
    return float(stake) if stake >= cfg.min_stake else 0.0


@dataclass
class StakeRecommendation:
    match_id: str
    market: str
    selection: str
    odds: float
    p_model_paid: float
    edge: float
    ev: float
    kelly_full: float
    kelly_fraction_used: float
    stake_frac: float
    stake: float
    expected_log_growth: float


def allocate_stakes(bets: list[ValueBet], bankroll: float, cfg: StakingConfig | None = None) -> pd.DataFrame:
    """Turn flagged value bets into stakes.

    Rules: flagged and EV > 0; one selection per (match, market); per-bet cap;
    proportional scale-down when total exposure exceeds `max_total_frac`.
    """
    cfg = cfg or StakingConfig()
    candidates = [b for b in bets if b.flagged and b.ev > 0]
    best: dict[tuple[str, str], ValueBet] = {}
    for b in candidates:
        key = (b.match_id, b.market)
        if key not in best or b.ev > best[key].ev:
            best[key] = b
    rows = []
    for b in best.values():
        f_full = kelly_fraction(b.p_model_paid, b.odds)
        f = min(f_full * cfg.kelly_fraction, cfg.max_stake_frac)
        rows.append((b, f_full, f))
    total = sum(f for _, _, f in rows)
    scale = min(1.0, cfg.max_total_frac / total) if total > 0 else 1.0
    out = []
    for b, f_full, f in rows:
        f_used = f * scale
        stake = bankroll * f_used
        if cfg.round_to > 0:
            stake = float(np.floor(stake / cfg.round_to) * cfg.round_to)
        if stake < cfg.min_stake or stake <= 0:
            continue
        out.append(
            StakeRecommendation(
                match_id=b.match_id,
                market=b.market,
                selection=b.selection,
                odds=b.odds,
                p_model_paid=b.p_model_paid,
                edge=b.edge,
                ev=b.ev,
                kelly_full=f_full,
                kelly_fraction_used=cfg.kelly_fraction,
                stake_frac=stake / bankroll if bankroll > 0 else 0.0,
                stake=stake,
                expected_log_growth=expected_log_growth(stake / bankroll, b.p_model_paid, b.odds) if bankroll > 0 else 0.0,
            )
        )
    cols = list(StakeRecommendation.__dataclass_fields__)
    return pd.DataFrame([asdict(r) for r in out], columns=cols).sort_values("ev", ascending=False).reset_index(drop=True)
