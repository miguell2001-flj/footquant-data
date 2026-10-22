"""League-separated Elo ratings (Phase 1), updated after every match.

Expected home score: E_h = 1 / (1 + 10 ** ((R_a - (R_h + HFA)) / 400)).
Update: R_h += K * G * (S_h - E_h), R_a -= the same, where S_h is the realised
result (1 / 0.5 / 0) or the xG-implied expected points share, and G the
margin-of-victory multiplier. Ratings are keyed by (league, team), so a club's
Champions League rating is separate from its domestic one. Hvattum & Arntzen
(2010, Int. J. Forecasting 26) show goal-based Elo differences are a strong
1X2 covariate, which is how `elo_diff` is used here: as a feature for the
boosted layer, not as the probability model itself.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..config import EloConfig


def mov_multiplier(goal_diff: int) -> float:
    d = abs(int(goal_diff))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def xg_result_share(home_xg: float, away_xg: float, max_goals: int = 12) -> float:
    """P(X > Y) + 0.5 P(X = Y) with X ~ Pois(home_xg), Y ~ Pois(away_xg)."""
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, max(home_xg, 1e-6))
    pa = poisson.pmf(k, max(away_xg, 1e-6))
    joint = np.outer(ph, pa)
    joint /= joint.sum()
    return float(np.tril(joint, -1).sum() + 0.5 * np.trace(joint))


class LeagueElo:
    def __init__(self, cfg: EloConfig | None = None) -> None:
        self.cfg = cfg or EloConfig()
        self.ratings: dict[tuple[str, str], float] = defaultdict(lambda: self.cfg.init_rating)

    def rating(self, league: str, team: str) -> float:
        return self.ratings[(league, team)]

    def expected_home(self, r_home: float, r_away: float, neutral: bool = False) -> float:
        hfa = 0.0 if neutral else self.cfg.home_adv
        return 1.0 / (1.0 + 10 ** ((r_away - (r_home + hfa)) / 400.0))

    def update(
        self,
        league: str,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        home_xg: float | None = None,
        away_xg: float | None = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Apply one result; returns the (pre-match home, pre-match away) ratings."""
        r_h, r_a = self.rating(league, home), self.rating(league, away)
        e_h = self.expected_home(r_h, r_a, neutral)
        if self.cfg.result_from == "xg" and home_xg is not None and not np.isnan(home_xg):
            s_h = xg_result_share(home_xg, away_xg)
        else:
            s_h = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        g = mov_multiplier(home_goals - away_goals) if self.cfg.mov_multiplier else 1.0
        delta = self.cfg.k_factor * g * (s_h - e_h)
        self.ratings[(league, home)] = r_h + delta
        self.ratings[(league, away)] = r_a - delta
        return r_h, r_a

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Replay a date-sorted match table; return strictly pre-match ratings per match.

        Columns: match_id, elo_home_pre, elo_away_pre, elo_diff, elo_expected_home.
        """
        rows = []
        neutral_col = matches["neutral"].astype(bool).values if "neutral" in matches.columns else np.zeros(len(matches), bool)
        for (mid, lg, h, a, hg, ag, hx, ax), neu in zip(
            matches[["match_id", "league", "home", "away", "home_goals", "away_goals", "home_xg", "away_xg"]].itertuples(index=False),
            neutral_col,
        ):
            r_h, r_a = self.rating(lg, h), self.rating(lg, a)
            rows.append(
                {
                    "match_id": mid,
                    "elo_home_pre": r_h,
                    "elo_away_pre": r_a,
                    "elo_diff": r_h - r_a,
                    "elo_expected_home": self.expected_home(r_h, r_a, neu),
                }
            )
            self.update(lg, h, a, int(hg), int(ag), hx, ax, neu)
        return pd.DataFrame(rows)

    def predict_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Current ratings for upcoming fixtures (no update)."""
        rows = []
        for mid, lg, h, a in fixtures[["match_id", "league", "home", "away"]].itertuples(index=False):
            r_h, r_a = self.rating(lg, h), self.rating(lg, a)
            rows.append({"match_id": mid, "elo_home_pre": r_h, "elo_away_pre": r_a, "elo_diff": r_h - r_a,
                         "elo_expected_home": self.expected_home(r_h, r_a)})
        return pd.DataFrame(rows)
