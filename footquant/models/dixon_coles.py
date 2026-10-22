"""Dixon-Coles engine (Phase 2) - maximum likelihood via scipy.optimize.

Model
-----
For match k between home team i and away team j:

    X_k ~ Poisson(lambda_k),   log lambda_k = c + h + att_i + def_j
    Y_k ~ Poisson(mu_k),       log mu_k     = c     + att_j + def_i

with sum(att) = sum(def) = 0 for identifiability (c is the intercept, h the home
advantage on the log scale). The joint pmf is multiplied by the Dixon-Coles
dependence factor on the four low-score cells:

    tau(0,0) = 1 - lambda*mu*rho      tau(1,0) = 1 + mu*rho
    tau(0,1) = 1 + lambda*rho         tau(1,1) = 1 - rho          tau = 1 otherwise

which still sums to one over the grid. rho < 0 inflates 0-0 and 1-1 and deflates
1-0 / 0-1 - the direction found empirically in Dixon & Coles (1997, J. R. Stat.
Soc. C 46(2):265-280).

Estimation
----------
The parameters maximise the exponentially time-weighted log-likelihood

    L(theta) = sum_k phi(t_ref - t_k) [ log tau(x_k, y_k; lambda_k, mu_k, rho)
                                       + log Pois(yh_k; lambda_k) + log Pois(ya_k; mu_k) ]
    phi(d) = exp(-xi * d),  d in days

where (x_k, y_k) are observed goals - used inside tau, because the low-score
dependence is an empirical property of goals - and (yh_k, ya_k) are the fit
targets, a blend w * xG + (1 - w) * goals. With w > 0 the target is non-integer
and the Poisson term uses its continuous extension  y log m - m - lgamma(y + 1),
so the objective is a quasi-likelihood (ratio w is a tuning parameter, validated
out-of-sample like xi). xi cannot be estimated by the same likelihood (it would
go to zero); it is tuned by out-of-sample RPS - see backtest.walkforward.tune_xi.

The gradient is analytic (fits in tens of milliseconds), which is what makes
weekly refits across several seasons affordable in the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from ..config import ModelConfig
from ..data.schema import validate_matches
from .score_matrix import market_probs, score_matrix


def time_weights(dates: pd.Series, ref_date: pd.Timestamp, xi: float) -> np.ndarray:
    """phi(t) = exp(-xi * days before ref_date)."""
    days = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.total_seconds().values / 86400.0
    days = np.clip(days, 0.0, None)
    return np.exp(-xi * days)


def tau(xh, xa, lam, mu, rho):
    """Dixon-Coles dependence factor, vectorised over matches."""
    xh = np.asarray(xh)
    xa = np.asarray(xa)
    lam = np.asarray(lam, float)
    mu = np.asarray(mu, float)
    out = np.ones(np.broadcast(xh, xa, lam, mu).shape)
    m00 = (xh == 0) & (xa == 0)
    m01 = (xh == 0) & (xa == 1)
    m10 = (xh == 1) & (xa == 0)
    m11 = (xh == 1) & (xa == 1)
    out = np.where(m00, 1.0 - lam * mu * rho, out)
    out = np.where(m01, 1.0 + lam * rho, out)
    out = np.where(m10, 1.0 + mu * rho, out)
    out = np.where(m11, 1.0 - rho, out)
    return out


@dataclass
class DCParams:
    """Fitted Dixon-Coles parameters for one league."""

    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_adv: float
    rho: float
    xi: float
    xg_weight: float
    fitted_at: pd.Timestamp
    n_matches: int
    loglik: float
    converged: bool
    unknown_team_prior: tuple[float, float] = (0.0, 0.0)
    leagues: list[str] | None = None
    """Joint (cross-league) fit only: league names and their strength offsets."""
    league_offset: np.ndarray | None = None
    team_league: dict[str, str] | None = None
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.teams)}
        self._league_index = {l: i for i, l in enumerate(self.leagues)} if self.leagues else {}

    def _offset(self, team: str) -> float:
        if not self.leagues or self.team_league is None:
            return 0.0
        i = self._league_index.get(self.team_league.get(team, ""))
        return 0.0 if i is None else float(self.league_offset[i])

    def _att_def(self, team: str) -> tuple[float, float]:
        i = self.index.get(team)
        if i is None:
            return self.unknown_team_prior
        return float(self.attack[i]), float(self.defence[i])

    def rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """(lambda, mu) expected goals for a fixture."""
        att_h, def_h = self._att_def(home)
        att_a, def_a = self._att_def(away)
        h = 0.0 if neutral else self.home_adv
        g = self._offset(home) - self._offset(away)  # league strength gap (joint fit), 0 within a league
        lam = float(np.exp(self.intercept + h + att_h + def_a + g))
        mu = float(np.exp(self.intercept + att_a + def_h - g))
        return lam, mu

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence}).sort_values(
            "attack", ascending=False
        )

    def to_theta(self) -> np.ndarray:
        n = len(self.teams)
        theta = np.concatenate([self.attack[: n - 1], self.defence[: n - 1], [self.intercept, self.home_adv, self.rho]])
        if self.leagues:
            theta = np.concatenate([theta, self.league_offset[:-1]])
        return theta

    def league_table(self) -> pd.DataFrame:
        if not self.leagues:
            return pd.DataFrame(columns=["league", "offset", "goal_ratio_vs_mean"])
        return pd.DataFrame({"league": self.leagues, "offset": self.league_offset, "goal_ratio_vs_mean": np.exp(self.league_offset)})


class DixonColesModel:
    """One league's Dixon-Coles model (fit per league, `league-separated`), or a
    joint multi-league model with league-strength offsets when `fit` is given a
    team -> league map (for Champions League / Europa League pricing)."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self.params: DCParams | None = None

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _unpack(theta: np.ndarray, n: int):
        att = np.empty(n)
        att[:-1] = theta[: n - 1]
        att[-1] = -theta[: n - 1].sum()
        dfc = np.empty(n)
        dfc[:-1] = theta[n - 1 : 2 * n - 2]
        dfc[-1] = -theta[n - 1 : 2 * n - 2].sum()
        c, h, rho = theta[2 * n - 2], theta[2 * n - 1], theta[2 * n]
        return att, dfc, c, h, rho

    @staticmethod
    def _objective(theta, n, hi, ai, xh, xa, yh, ya, w, ridge, neutral, li=None, m=0):
        """Negative weighted log-likelihood (without lgamma constants) and its gradient.

        With `li` (league index per team) and `m` leagues, theta carries m - 1
        extra league-strength offsets g (sum-to-zero) entering as
        log lambda += g[L(home)] - g[L(away)], log mu += g[L(away)] - g[L(home)].
        """
        att, dfc, c, h, rho = DixonColesModel._unpack(theta[: 2 * n + 1], n)
        loglam = c + h * (1.0 - neutral) + att[hi] + dfc[ai]
        logmu = c + att[ai] + dfc[hi]
        if li is not None and m > 1:
            g = np.empty(m)
            g[:-1] = theta[2 * n + 1 :]
            g[-1] = -g[:-1].sum()
            gap = g[li[hi]] - g[li[ai]]
            loglam = loglam + gap
            logmu = logmu - gap
        lam = np.exp(loglam)
        mu = np.exp(logmu)

        m00 = (xh == 0) & (xa == 0)
        m01 = (xh == 0) & (xa == 1)
        m10 = (xh == 1) & (xa == 0)
        m11 = (xh == 1) & (xa == 1)
        t = np.ones_like(lam)
        dt_dlam = np.zeros_like(lam)
        dt_dmu = np.zeros_like(lam)
        dt_drho = np.zeros_like(lam)
        t[m00] = 1.0 - lam[m00] * mu[m00] * rho
        dt_dlam[m00] = -mu[m00] * rho
        dt_dmu[m00] = -lam[m00] * rho
        dt_drho[m00] = -lam[m00] * mu[m00]
        t[m01] = 1.0 + lam[m01] * rho
        dt_dlam[m01] = rho
        dt_drho[m01] = lam[m01]
        t[m10] = 1.0 + mu[m10] * rho
        dt_dmu[m10] = rho
        dt_drho[m10] = mu[m10]
        t[m11] = 1.0 - rho
        dt_drho[m11] = -1.0

        valid = t > 1e-10
        t_c = np.where(valid, t, 1e-10)
        inv_t = np.where(valid, 1.0 / t_c, 0.0)

        ll = yh * loglam - lam + ya * logmu - mu + np.log(t_c)
        nll = -float(np.sum(w * ll)) + ridge * float(att @ att + dfc @ dfc)

        dlam = w * (yh - lam + dt_dlam * inv_t * lam)  # d ll / d loglam
        dmu = w * (ya - mu + dt_dmu * inv_t * mu)  # d ll / d logmu
        drho = float(np.sum(w * dt_drho * inv_t))
        g_att = np.bincount(hi, weights=dlam, minlength=n) + np.bincount(ai, weights=dmu, minlength=n)
        g_def = np.bincount(ai, weights=dlam, minlength=n) + np.bincount(hi, weights=dmu, minlength=n)
        g_c = float(dlam.sum() + dmu.sum())
        g_h = float(np.sum(dlam * (1.0 - neutral)))
        grad = np.concatenate(
            [
                -(g_att[:-1] - g_att[-1]) + 2.0 * ridge * (att[:-1] - att[-1]),
                -(g_def[:-1] - g_def[-1]) + 2.0 * ridge * (dfc[:-1] - dfc[-1]),
                [-g_c, -g_h, -drho],
            ]
        )
        if li is not None and m > 1:
            lh, la = li[hi], li[ai]
            g_g = (np.bincount(lh, weights=dlam, minlength=m) - np.bincount(la, weights=dlam, minlength=m)
                   + np.bincount(la, weights=dmu, minlength=m) - np.bincount(lh, weights=dmu, minlength=m))
            grad = np.concatenate([grad, -(g_g[:-1] - g_g[-1])])
        return nll, grad

    def fit(
        self,
        matches: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
        warm_start: bool = True,
        team_league: dict[str, str] | None = None,
    ) -> DCParams:
        """Maximum-likelihood fit on all rows with date <= as_of (default: the last match).

        Time weights decay from `as_of`. The caller controls what the model may
        see: in a walk-forward backtest pass only matches dated before the
        prediction window and `as_of` = window start.

        `team_league` (team -> domestic league) switches on the joint
        cross-league fit: one attack/defence pair per team as before, plus one
        strength offset per league identified by the inter-league fixtures
        (Champions League, Europa League...) in `matches`. Within a league the
        offsets cancel, so domestic predictions are unchanged in form; a
        cross-league fixture gets the league gap on both rates.
        """
        df = validate_matches(matches)
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
        df = df[df["date"] <= as_of]
        if len(df) < 20:
            raise ValueError("need at least 20 matches to fit")
        teams = sorted(set(df["home"]) | set(df["away"]))
        n = len(teams)
        index = {t: i for i, t in enumerate(teams)}
        hi = df["home"].map(index).values.astype(int)
        ai = df["away"].map(index).values.astype(int)
        xh = df["home_goals"].values.astype(int)
        xa = df["away_goals"].values.astype(int)

        w_xg = self.cfg.xg_weight
        have_xg = df["home_xg"].notna() & df["away_xg"].notna()
        if w_xg > 0 and not have_xg.all():
            w_xg_row = np.where(have_xg.values, w_xg, 0.0)  # rows without xG fall back to goals
        else:
            w_xg_row = np.full(len(df), w_xg)
        yh = w_xg_row * np.nan_to_num(df["home_xg"].values) + (1 - w_xg_row) * xh
        ya = w_xg_row * np.nan_to_num(df["away_xg"].values) + (1 - w_xg_row) * xa

        w = time_weights(df["date"], as_of, self.cfg.xi)
        neutral = df["neutral"].astype(float).values if "neutral" in df.columns else np.zeros(len(df))

        leagues, li, m = None, None, 0
        if team_league is not None:
            missing = [t for t in teams if t not in team_league]
            if missing:
                raise ValueError(f"team_league has no entry for {missing[:5]}{'...' if len(missing) > 5 else ''}")
            leagues = sorted(set(team_league[t] for t in teams))
            m = len(leagues)
            lidx = {l: i for i, l in enumerate(leagues)}
            li = np.array([lidx[team_league[t]] for t in teams], dtype=int)

        theta0 = None
        if warm_start and self.params is not None and self.params.teams == teams and (self.params.leagues or None) == leagues:
            theta0 = self.params.to_theta()
        if theta0 is None:
            mean_h = max(float(np.average(yh, weights=w)), 1e-3)
            mean_a = max(float(np.average(ya, weights=w)), 1e-3)
            theta0 = np.concatenate([np.zeros(2 * n - 2), [np.log(mean_a), np.log(mean_h / mean_a), 0.0], np.zeros(max(m - 1, 0))])

        bounds = [(-3.0, 3.0)] * (2 * n - 2) + [(-3.0, 3.0), (-1.0, 1.5), tuple(self.cfg.rho_bounds)] + [(-2.0, 2.0)] * max(m - 1, 0)
        res = minimize(
            self._objective,
            theta0,
            args=(n, hi, ai, xh, xa, yh, ya, w, self.cfg.ridge, neutral, li, m),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.cfg.max_iter},
        )
        att, dfc, c, h, rho = self._unpack(res.x[: 2 * n + 1], n)
        offsets = None
        if m > 1:
            offsets = np.empty(m)
            offsets[:-1] = res.x[2 * n + 1 :]
            offsets[-1] = -offsets[:-1].sum()
        const = -float(np.sum(w * (gammaln(yh + 1) + gammaln(ya + 1))))
        self.params = DCParams(
            teams=teams,
            attack=att,
            defence=dfc,
            intercept=float(c),
            home_adv=float(h),
            rho=float(rho),
            xi=self.cfg.xi,
            xg_weight=w_xg,
            fitted_at=as_of,
            n_matches=int(len(df)),
            loglik=-(float(res.fun) - self.cfg.ridge * float(att @ att + dfc @ dfc)) + const,
            converged=bool(res.success),
            unknown_team_prior=self.cfg.unknown_team_prior,
            leagues=leagues,
            league_offset=offsets,
            team_league=dict(team_league) if team_league is not None else None,
        )
        return self.params

    # --------------------------------------------------------------- predicting
    def _require(self) -> DCParams:
        if self.params is None:
            raise RuntimeError("model is not fitted")
        return self.params

    def predict_rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        return self._require().rates(home, away, neutral)

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        lam, mu = self.predict_rates(home, away, neutral)
        return score_matrix(lam, mu, self._require().rho, self.cfg.max_goals)

    def predict(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        """Market probabilities for one fixture: 1X2, BTTS, O/U 2.5, plus lambda/mu/rho."""
        lam, mu = self.predict_rates(home, away, neutral)
        p = market_probs(score_matrix(lam, mu, self._require().rho, self.cfg.max_goals))
        p.update({"lam": lam, "mu": mu, "rho": self._require().rho})
        return p

    def predict_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Vectorised `predict` over a fixture table (match_id, home, away[, neutral])."""
        neutral = fixtures["neutral"].astype(bool).values if "neutral" in fixtures.columns else [False] * len(fixtures)
        rows = []
        for (mid, h, a), neu in zip(fixtures[["match_id", "home", "away"]].itertuples(index=False), neutral):
            rec = {"match_id": mid, "home": h, "away": a}
            rec.update(self.predict(h, a, bool(neu)))
            rows.append(rec)
        return pd.DataFrame(rows)
