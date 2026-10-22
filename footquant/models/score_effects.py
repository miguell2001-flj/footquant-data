"""Score- and time-dependent goal intensities for the early-payout path model.

The independent-Poisson path (``score_matrix.early_payout_probs``) assumes a
constant scoring rate. Real matches are not like that: scoring rises through
the match and depends on the current score - a trailing side pushes, a leading
side sits back (Dixon & Robinson 1998, *J. R. Stat. Soc. D* 47(3):523-538, who
model a match as a birth process with time- and score-dependent rates). The
effect matters here because "led by two, did not win" is exactly the path that
score effects make more likely. Across the big-5 leagues 2016-25 the constant-
rate model under-predicts away rescues by 0.2-0.8 pp.

Model
-----
In minute m (0..95, i.e. 90 minutes plus ~6 of stoppage) with the scoring
side's current lead s (clipped to -2..+2) the goal intensity is

    rate_home(m, s) = lam / 96 * F_home(bin(m), s)
    rate_away(m, s) = mu  / 96 * F_away(bin(m), s)

where F = exp(c + a_time[bin] + b_state[s]) is fitted by Poisson regression on
a minute-by-minute exposure table (one row per match-minute-side, offset
log(lam/96) or log(mu/96) from a Dixon-Coles fit of the same matches). The
time profile and the state effect are separate for the home and away processes.

`ScoreEffectModel.simulate` runs the inhomogeneous process minute by minute and
returns the same keys as `early_payout_probs` (p_home_bonus, p_away_bonus, ...)
plus the implied 1X2 probabilities. A model fitted on the five leagues
2014/15-2024/25 ships as the package default (`default_score_effects()`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES = 96
TIME_BINS = (0, 15, 30, 45, 60, 75, 90, 96)
STATES = (-2, -1, 0, 1, 2)
_DEFAULT_PATH = Path(__file__).with_name("score_effects_default.json")


def _bin_index(minute: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(TIME_BINS[1:]), minute, side="right"), 0, len(TIME_BINS) - 2)


@dataclass
class ScoreEffectModel:
    """F[side][bin, state] multipliers on the per-minute rate."""

    f_home: list[list[float]]
    f_away: list[list[float]]
    n_matches: int = 0
    note: str = ""
    _fh: np.ndarray = field(init=False, repr=False)
    _fa: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fh = np.asarray(self.f_home, float)
        self._fa = np.asarray(self.f_away, float)
        assert self._fh.shape == (len(TIME_BINS) - 1, len(STATES)) == self._fa.shape

    # ---------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"f_home": self.f_home, "f_away": self.f_away, "n_matches": self.n_matches, "note": self.note}, fh, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreEffectModel":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(d["f_home"], d["f_away"], d.get("n_matches", 0), d.get("note", ""))

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for side, f in (("home", self._fh), ("away", self._fa)):
            for b in range(len(TIME_BINS) - 1):
                for j, s in enumerate(STATES):
                    rows.append({"side": side, "minutes": f"{TIME_BINS[b]}-{TIME_BINS[b + 1]}", "lead": s, "factor": float(f[b, j])})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- simulation
    def simulate(self, lam: float, mu: float, lead: int = 2, n_sims: int = 50_000, seed: int | None = None) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        diff = np.zeros(n_sims, dtype=np.int16)
        hmax = np.zeros(n_sims, dtype=np.int16)
        amax = np.zeros(n_sims, dtype=np.int16)
        hg = np.zeros(n_sims, dtype=np.int16)
        ag = np.zeros(n_sims, dtype=np.int16)
        base_h, base_a = lam / MINUTES, mu / MINUTES
        for m in range(MINUTES):
            b = int(_bin_index(np.array([m]))[0])
            s_home = np.clip(diff, -2, 2) + 2  # column index of the home side's lead
            s_away = np.clip(-diff, -2, 2) + 2
            p_h = -np.expm1(-base_h * self._fh[b, s_home])
            p_a = -np.expm1(-base_a * self._fa[b, s_away])
            u = rng.random(n_sims)
            goal_h = u < p_h
            goal_a = (~goal_h) & (u < p_h + p_a)  # at most one goal per minute-step
            hg += goal_h
            ag += goal_a
            diff = (hg - ag).astype(np.int16)
            hmax = np.maximum(hmax, diff)
            amax = np.maximum(amax, -diff)
        home_win = diff > 0
        away_win = diff < 0
        home_reached = hmax >= lead
        away_reached = amax >= lead
        return {
            "p_home_reached": float(home_reached.mean()),
            "p_away_reached": float(away_reached.mean()),
            "p_home_bonus": float((home_reached & ~home_win).mean()),
            "p_away_bonus": float((away_reached & ~away_win).mean()),
            "p_home_win_sim": float(home_win.mean()),
            "p_draw_sim": float((diff == 0).mean()),
            "p_away_win_sim": float(away_win.mean()),
            "exp_home_goals_sim": float(hg.mean()),
            "exp_away_goals_sim": float(ag.mean()),
        }


def minute_table(matches: pd.DataFrame, goals: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, minute, side): exposure, offset rate, lead state, goals.

    `goals`: match_id, team, minute for every goal (own goals credited to the
    scoring side, as `load_understat_shots` returns them). `rates`: match_id,
    lam, mu (the baseline rates the multipliers are relative to).
    """
    m = matches.set_index("match_id")
    r = rates.set_index("match_id")
    ids = [i for i in m.index if i in r.index]
    g = goals[goals["match_id"].isin(ids)].copy()
    g["minute"] = np.clip(g["minute"].astype(float).values, 0, MINUTES - 1).astype(int)
    g["is_home"] = g["team"].values == m.loc[g["match_id"], "home"].values
    # per-match minute grids
    n = len(ids)
    hg = np.zeros((n, MINUTES), np.int16)
    ag = np.zeros((n, MINUTES), np.int16)
    idx = {mid: k for k, mid in enumerate(ids)}
    rows = g["match_id"].map(idx).values
    np.add.at(hg, (rows[g["is_home"].values], g.loc[g["is_home"], "minute"].values), 1)
    np.add.at(ag, (rows[~g["is_home"].values], g.loc[~g["is_home"], "minute"].values), 1)
    before = np.cumsum(hg - ag, axis=1) - (hg - ag)  # lead before the minute's goals
    minute = np.tile(np.arange(MINUTES), n)
    out = pd.DataFrame(
        {
            "match_id": np.repeat(ids, MINUTES),
            "minute": minute,
            "tbin": _bin_index(minute),
            "lead_home": np.clip(before.ravel(), -2, 2),
            "y_home": hg.ravel(),
            "y_away": ag.ravel(),
            "off_home": np.repeat(r.loc[ids, "lam"].values / MINUTES, MINUTES),
            "off_away": np.repeat(r.loc[ids, "mu"].values / MINUTES, MINUTES),
        }
    )
    return out


def fit_score_effects(matches: pd.DataFrame, goals: pd.DataFrame, rates: pd.DataFrame, note: str = "") -> ScoreEffectModel:
    """Poisson GLM of goals per minute on time bin x lead state, per side, with
    the baseline per-minute rate as offset (fitted as y/offset with weight offset)."""
    from sklearn.linear_model import PoissonRegressor

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    tab = minute_table(matches, goals, rates)
    nb, ns = len(TIME_BINS) - 1, len(STATES)
    ref = STATES.index(0)  # level score is the reference state -> full-rank design

    def design(tbin, lead):
        X = np.zeros((len(tbin), nb + ns - 1))
        X[np.arange(len(tbin)), tbin] = 1.0
        col = np.asarray(lead) + 2
        col = np.where(col > ref, col - 1, col)
        mask = np.asarray(lead) != 0
        X[np.arange(len(tbin))[mask], nb + col[mask]] = 1.0
        return X

    out = {}
    for side, lead_col_sign in (("home", 1), ("away", -1)):
        lead = np.clip(tab["lead_home"].values * lead_col_sign, -2, 2)
        X = design(tab["tbin"].values, lead)
        y = tab[f"y_{side}"].values.astype(float)
        off = tab[f"off_{side}"].values
        reg = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=10_000, tol=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            reg.fit(X, y / off, sample_weight=off)
        coef = reg.coef_
        b = np.insert(coef[nb:], ref, 0.0)
        F = np.exp(coef[:nb][:, None] + b[None, :])  # bin x state
        out[side] = F
    return ScoreEffectModel(out["home"].tolist(), out["away"].tolist(), n_matches=int(tab["match_id"].nunique()), note=note)


_DEFAULT: ScoreEffectModel | None = None


def default_score_effects() -> ScoreEffectModel:
    """The model shipped with the package (big-5 leagues 2014/15-2024/25)."""
    global _DEFAULT
    if _DEFAULT is None:
        if not _DEFAULT_PATH.exists():
            raise FileNotFoundError("no default score-effects model file; fit one with fit_score_effects and save it")
        _DEFAULT = ScoreEffectModel.load(_DEFAULT_PATH)
    return _DEFAULT
