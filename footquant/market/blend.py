"""Combining model and market probabilities - the market as a prior.

A sharp closing (or opening) line is the best single forecast available, so
the practical question is not "model or market" but "does the model carry
information the market lacks, and how much weight does it deserve?".

Log-linear pool (one parameter, interpretable):

    p_i  ∝  p_model,i ** w  *  p_market,i ** (1 - w),      w in [0, 1]

w is fitted by maximum likelihood on past matches (`fit_pool_weight`). This is
Granger-Ramanathan style forecast encompassing in probability space: w -> 0
means the market encompasses the model; w materially above 0, out of sample,
means the model adds information that the price has not absorbed. Only that
information can be an edge; everything else in the model's disagreement with
the market is estimation noise, which is what the EPL backtest in the README
shows up as negative closing-line value.

Multinomial stacking (`MultinomialStacker`): a multinomial logistic regression
on the six log-probabilities, which also re-calibrates the pool. Fit on
earlier seasons, apply to later ones - never in-sample.

Market-implied rates (`market_implied_rates`): invert the Dixon-Coles score
matrix to the (lambda, mu) that reproduce the market's 1X2 (and, if given,
over/under 2.5) probabilities. These rates let the market enter the rate-level
architecture directly: the boosted layer can take them as its base rate, or the
model's rates can be shrunk toward them before every market is priced from one
consistent score matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ..models.score_matrix import market_probs, score_matrix

_EPS = 1e-9


def log_linear_pool(p_model, p_market, w: float) -> np.ndarray:
    pm = np.clip(np.asarray(p_model, float), _EPS, 1.0)
    pk = np.clip(np.asarray(p_market, float), _EPS, 1.0)
    logp = w * np.log(pm) + (1.0 - w) * np.log(pk)
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def _nll(p, outcomes) -> float:
    o = np.asarray(outcomes).astype(int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(o)), o], _EPS, 1.0))))


def fit_pool_weight(p_model, p_market, outcomes) -> tuple[float, dict[str, float]]:
    """Maximum-likelihood pool weight on the model, with the log loss of the
    model alone, the market alone and the pool (all in-sample on the data given)."""
    pm = np.asarray(p_model, float)
    pk = np.asarray(p_market, float)
    res = minimize_scalar(lambda w: _nll(log_linear_pool(pm, pk, w), outcomes), bounds=(0.0, 1.0), method="bounded")
    w = float(res.x)
    return w, {
        "log_loss_model": _nll(pm, outcomes),
        "log_loss_market": _nll(pk, outcomes),
        "log_loss_pool": float(res.fun),
    }


class MultinomialStacker:
    """Multinomial logistic regression on [log p_model (3), log p_market (3)]."""

    def __init__(self, C: float = 10.0) -> None:
        from sklearn.linear_model import LogisticRegression

        self._est = LogisticRegression(C=C, max_iter=2000)

    @staticmethod
    def _X(p_model, p_market) -> np.ndarray:
        pm = np.clip(np.asarray(p_model, float), _EPS, 1.0)
        pk = np.clip(np.asarray(p_market, float), _EPS, 1.0)
        return np.hstack([np.log(pm), np.log(pk)])

    def fit(self, p_model, p_market, outcomes) -> "MultinomialStacker":
        self._est.fit(self._X(p_model, p_market), np.asarray(outcomes).astype(int))
        return self

    def predict(self, p_model, p_market) -> np.ndarray:
        return self._est.predict_proba(self._X(p_model, p_market))

    @property
    def coef_(self) -> np.ndarray:
        return self._est.coef_


def pool_with_market(
    probs: dict,
    odds_1x2: list[float],
    w: float,
    over25_fair: float | None = None,
    max_goals: int = 8,
    n_sims: int = 50_000,
    seed: int = 0,
    devig_method: str = "shin",
    value_cfg=None,
) -> tuple[dict, dict]:
    """Pool a fixture's model output (`DixonColesModel.predict` dict with lam,
    mu, rho, p_home/draw/away) with a 1X2 price vector.

    Returns (pooled_probs, early_payout): the pooled dict carries the pooled
    1X2 probabilities, the pooled rates lam/mu (geometric blend of model and
    market-implied rates), the market's fair probabilities (m_home/m_draw/m_away)
    and the market-implied rates; `early_payout` is `early_payout_probs` on the
    pooled rates. BTTS / O-U probabilities are re-derived from the pooled rates.
    """
    from .devig import devig  # local import: devig does not depend on blend

    pk = devig(odds_1x2, devig_method)
    lam_k, mu_k = market_implied_rates(pk[0], pk[1], pk[2], over25_fair, rho=probs["rho"], max_goals=10)
    pm = np.array([probs["p_home"], probs["p_draw"], probs["p_away"]])
    pool = log_linear_pool(pm[None, :], pk[None, :], w)[0]
    lam_p = probs["lam"] ** w * lam_k ** (1 - w)
    mu_p = probs["mu"] ** w * mu_k ** (1 - w)
    out = dict(probs)
    out.update(market_probs(score_matrix(lam_p, mu_p, probs["rho"], max_goals)))  # totals / BTTS from pooled rates
    out.update({"p_home": float(pool[0]), "p_draw": float(pool[1]), "p_away": float(pool[2]),
                "lam": float(lam_p), "mu": float(mu_p), "lam_model": probs["lam"], "mu_model": probs["mu"],
                "lam_market": lam_k, "mu_market": mu_k, "m_home": float(pk[0]), "m_draw": float(pk[1]), "m_away": float(pk[2]),
                "p_model_home": float(pm[0]), "p_model_draw": float(pm[1]), "p_model_away": float(pm[2]), "pool_weight": w})
    if value_cfg is None:
        from ..models.score_matrix import early_payout_probs

        ep = early_payout_probs(lam_p, mu_p, 2, n_sims, seed=seed)
    else:
        from ..models.early_payout import simulate_early_payout

        ep = simulate_early_payout(lam_p, mu_p, value_cfg, seed=seed, n_sims=n_sims)
    return out, ep


def market_implied_rates(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over25: float | None = None,
    rho: float = -0.1,
    max_goals: int = 10,
) -> tuple[float, float]:
    """(lambda, mu) whose Dixon-Coles score matrix reproduces the market's
    de-vigged 1X2 probabilities (and over 2.5, when supplied) by least squares
    on log-odds. Without a totals price the solution is pinned by the 1X2
    shape only, which identifies lambda - mu well and lambda + mu less well."""
    target = np.array([p_home, p_away] + ([p_over25] if p_over25 is not None else []))

    def loss(theta):
        lam, mu = np.exp(theta)
        p = market_probs(score_matrix(lam, mu, rho, max_goals))
        got = np.array([p["p_home"], p["p_away"]] + ([p["p_over25"]] if p_over25 is not None else []))
        got = np.clip(got, 1e-6, 1 - 1e-6)
        return float(np.sum((np.log(got / (1 - got)) - np.log(target / (1 - target))) ** 2))

    res = minimize(loss, x0=np.log([1.4, 1.1]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
    lam, mu = np.exp(res.x)
    return float(lam), float(mu)
