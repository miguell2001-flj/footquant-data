"""Proper scoring rules for 1X2 forecasts (Phase 6).

With forecast vector p = (p_home, p_draw, p_away) and outcome indicator e:

    Brier      = sum_j (p_j - e_j)^2                       (0 best, 2 worst)
    Log loss   = -ln p_outcome                             (0 best)
    RPS        = 1/(r-1) sum_{i=1}^{r-1} ( sum_{j<=i} (p_j - e_j) )^2,  r = 3

RPS is the ranked probability score: it treats home/draw/away as ordered, so
predicting "home" when the result is a draw is penalised less than predicting
"away". Constantinou & Fenton (2012, J. Quant. Anal. Sports 8(1)) show it is
the appropriate criterion for football 1X2 models, which is why the time-decay
parameter xi is tuned on it. All three are averaged over matches.

Closing line value: CLV = odds_taken * p_fair_closing - 1, the expected return
of the bet priced at the de-vigged closing line. A positive mean CLV over
hundreds of bets is the earliest reliable evidence of real edge (the closing
line is the market's best estimate; P&L alone needs thousands of bets to
separate skill from variance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_INDEX = {"home": 0, "draw": 1, "away": 2}


def _onehot(outcomes) -> np.ndarray:
    o = np.asarray(outcomes).astype(int)
    e = np.zeros((len(o), 3))
    e[np.arange(len(o)), o] = 1.0
    return e


def outcome_index(home_goals, away_goals) -> np.ndarray:
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def brier_score(probs, outcomes) -> float:
    p = np.asarray(probs, float)
    return float(np.mean(np.sum((p - _onehot(outcomes)) ** 2, axis=1)))


def log_loss(probs, outcomes, eps: float = 1e-12) -> float:
    p = np.asarray(probs, float)
    o = np.asarray(outcomes).astype(int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(o)), o], eps, 1.0))))


def rps(probs, outcomes) -> float:
    p = np.asarray(probs, float)
    e = _onehot(outcomes)
    cum = np.cumsum(p - e, axis=1)[:, :-1]  # the last cumulative term is identically 0
    return float(np.mean(np.sum(cum**2, axis=1) / (p.shape[1] - 1)))


def evaluate_predictions(preds: pd.DataFrame) -> dict[str, float]:
    """Score a frame with p_home, p_draw, p_away, home_goals, away_goals."""
    probs = preds[["p_home", "p_draw", "p_away"]].values
    outcomes = outcome_index(preds["home_goals"].values, preds["away_goals"].values)
    return {
        "n": int(len(preds)),
        "brier": brier_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "rps": rps(probs, outcomes),
        "accuracy": float(np.mean(probs.argmax(axis=1) == outcomes)),
    }


def closing_line_value(odds_taken: float, p_fair_closing: float) -> float:
    return odds_taken * p_fair_closing - 1.0


def max_drawdown(bankroll_path) -> float:
    b = np.asarray(bankroll_path, float)
    peak = np.maximum.accumulate(b)
    return float(np.max((peak - b) / peak)) if len(b) else 0.0
