"""Score matrix -> market probabilities, and the early-payout path model (Phases 3 & 4).

P[x, y] is the Dixon-Coles probability of the home side scoring x and the away
side y, x, y = 0..max_goals (8 -> the 9x9 grid). Every market is a sum over
cells:

    P(home) = sum_{x>y} P[x,y]     P(draw) = sum_x P[x,x]     P(away) = sum_{x<y} P[x,y]
    P(BTTS)  = 1 - P[0,:].sum() - P[:,0].sum() + P[0,0]
    P(O2.5)  = sum_{x+y>=3} P[x,y]

Early payout: bet365 settles a pre-match Full Time Result bet as a winner the
moment the backed side leads by two goals. The payout probability is therefore

    P(paid) = P(win) + P(led by 2 at some point AND did not win)

The second term is path-dependent. Goals are modelled as a superposed Poisson
process at rate lambda + mu in which each goal is scored by the home side with
probability lambda / (lambda + mu) (thinning); the running goal difference is a
random walk whose maximum is tracked by Monte Carlo. The final-score marginals
of this process are exactly the independent Poissons, so the increment is
consistent with the rate model; the tau correction (which only re-weights the
0-0, 1-0, 0-1, 1-1 cells) is not applied to the path, which is why the result
is reported as an *increment* to be added to the tau-adjusted win probability.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 8) -> np.ndarray:
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, lam)
    pa = poisson.pmf(k, mu)
    P = np.outer(ph, pa)
    P[0, 0] *= 1.0 - lam * mu * rho
    P[0, 1] *= 1.0 + lam * rho
    P[1, 0] *= 1.0 + mu * rho
    P[1, 1] *= 1.0 - rho
    P = np.clip(P, 0.0, None)
    return P / P.sum()


def market_probs(P: np.ndarray) -> dict[str, float]:
    n = P.shape[0]
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    home = float(P[x > y].sum())
    draw = float(np.trace(P))
    away = float(P[x < y].sum())
    btts = float(1.0 - P[0, :].sum() - P[:, 0].sum() + P[0, 0])
    over25 = float(P[(x + y) >= 3].sum())
    most_likely = np.unravel_index(int(P.argmax()), P.shape)
    return {
        "p_home": home,
        "p_draw": draw,
        "p_away": away,
        "p_btts_yes": btts,
        "p_btts_no": 1.0 - btts,
        "p_over25": over25,
        "p_under25": 1.0 - over25,
        "exp_home_goals": float((P * x).sum()),
        "exp_away_goals": float((P * y).sum()),
        "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
    }


def early_payout_probs(
    lam: float,
    mu: float,
    lead: int = 2,
    n_sims: int = 200_000,
    seed: int | None = None,
    max_total_goals: int = 20,
) -> dict[str, float]:
    """Monte-Carlo estimate of the early-payout increments for both sides.

    Returns
      p_home_reached / p_away_reached : P(side led by `lead` at some point)
      p_home_bonus / p_away_bonus     : P(reached AND did not win) - add to P(win)
      p_home_win_sim / p_away_win_sim : simulated win probabilities (diagnostic;
                                        tau-free, so slightly off the DC values)
    """
    rng = np.random.default_rng(seed)
    total = np.minimum(rng.poisson(lam + mu, n_sims), max_total_goals)
    max_t = max(int(total.max()), 1)
    p_home = lam / (lam + mu)
    steps = np.where(rng.random((n_sims, max_t)) < p_home, 1, -1).astype(np.int8)
    mask = np.arange(max_t)[None, :] < total[:, None]
    path = np.cumsum(steps * mask, axis=1, dtype=np.int16)
    final = path[:, -1]
    hmax = np.maximum(path.max(axis=1), 0)
    amax = np.maximum(-path.min(axis=1), 0)
    home_reached = hmax >= lead
    away_reached = amax >= lead
    home_win = final > 0
    away_win = final < 0
    return {
        "p_home_reached": float(home_reached.mean()),
        "p_away_reached": float(away_reached.mean()),
        "p_home_bonus": float((home_reached & ~home_win).mean()),
        "p_away_bonus": float((away_reached & ~away_win).mean()),
        "p_home_win_sim": float(home_win.mean()),
        "p_away_win_sim": float(away_win.mean()),
    }
