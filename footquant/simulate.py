"""Synthetic league generator - used by the tests and the demo.

Generates a double round-robin league from known Dixon-Coles parameters so the
estimation code can be checked for parameter recovery, plus:
  * xG as a noisy measurement of the true scoring rate (lognormal noise),
  * goal timelines (so `home_max_lead` / `away_max_lead` are available for
    early-payout settlement),
  * a synthetic bookmaker whose prices are the true probabilities perturbed by
    pricing noise and inflated by a margin - so the value engine can be tested
    against a book that is sharp (noise 0) or beatable (noise > 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models.score_matrix import score_matrix


def simulate_league(
    n_teams: int = 20,
    n_seasons: int = 3,
    seed: int = 0,
    intercept: float = 0.10,
    home_adv: float = 0.25,
    rho: float = -0.10,
    att_sd: float = 0.25,
    def_sd: float = 0.20,
    xg_noise: float = 0.35,
    league: str = "SYN-League",
    start: str = "2023-08-12",
    max_goals: int = 10,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(n_teams)]
    att = rng.normal(0.0, att_sd, n_teams)
    att -= att.mean()
    dfc = rng.normal(0.0, def_sd, n_teams)
    dfc -= dfc.mean()

    rows = []
    t0 = pd.Timestamp(start)
    match_no = 0
    for season in range(n_seasons):
        season_label = f"{t0.year + season}-{str(t0.year + season + 1)[-2:]}"
        # circle-method schedule: n_teams - 1 rounds, then the reverse fixtures
        order = list(range(n_teams))
        rounds = []
        for _ in range(n_teams - 1):
            pairs = [(order[i], order[n_teams - 1 - i]) for i in range(n_teams // 2)]
            rounds.append(pairs)
            order = [order[0]] + [order[-1]] + order[1:-1]
        rounds += [[(b, a) for a, b in r] for r in rounds]
        for r_idx, pairs in enumerate(rounds):
            week = season * 52 + r_idx + (0 if r_idx < len(rounds) // 2 else 2)  # winter break
            for h, a in pairs:
                lam = np.exp(intercept + home_adv + att[h] + dfc[a])
                mu = np.exp(intercept + att[a] + dfc[h])
                P = score_matrix(lam, mu, rho, max_goals)
                flat = rng.choice(P.size, p=P.ravel())
                hg, ag = divmod(int(flat), max_goals + 1)
                # goal timeline: uniformly ordered goals -> running lead
                seq = np.array([1] * hg + [-1] * ag)
                rng.shuffle(seq)
                diff = np.cumsum(seq) if len(seq) else np.array([0])
                date = t0 + pd.Timedelta(days=7 * week) + pd.Timedelta(days=int(rng.integers(0, 3)), hours=int(rng.integers(12, 21)))
                rows.append(
                    {
                        "match_id": f"{league}|{match_no:05d}",
                        "date": date,
                        "league": league,
                        "season": season_label,
                        "home": teams[h],
                        "away": teams[a],
                        "home_goals": hg,
                        "away_goals": ag,
                        "home_xg": lam * np.exp(rng.normal(-xg_noise**2 / 2, xg_noise)),
                        "away_xg": mu * np.exp(rng.normal(-xg_noise**2 / 2, xg_noise)),
                        "home_max_lead": int(max(diff.max(), 0)),
                        "away_max_lead": int(max(-diff.min(), 0)),
                        "true_lam": lam,
                        "true_mu": mu,
                    }
                )
                match_no += 1
    matches = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    truth = {
        "teams": teams,
        "attack": att,
        "defence": dfc,
        "intercept": intercept,
        "home_adv": home_adv,
        "rho": rho,
    }
    return matches, truth


def synthetic_bookmaker(
    matches: pd.DataFrame,
    rho: float,
    margin: float = 0.05,
    noise_sd: float = 0.0,
    closing_noise_sd: float | None = None,
    seed: int = 1,
    bookmaker: str = "bet365",
    max_goals: int = 10,
) -> pd.DataFrame:
    """Price 1X2, BTTS and O/U 2.5 from the true rates with pricing noise + margin.

    noise_sd perturbs the *true* probabilities (logit scale) before the margin
    is applied; closing lines use `closing_noise_sd` (default: half the opening
    noise, i.e. the market sharpens toward kick-off). Returns an ODDS_COLUMNS frame.
    """
    rng = np.random.default_rng(seed)
    if closing_noise_sd is None:
        closing_noise_sd = noise_sd / 2
    rows = []
    for rec in matches.itertuples(index=False):
        P = score_matrix(rec.true_lam, rec.true_mu, rho, max_goals)
        x = np.arange(max_goals + 1)[:, None]
        y = np.arange(max_goals + 1)[None, :]
        markets = {
            "1X2": {"home": P[x > y].sum(), "draw": np.trace(P), "away": P[x < y].sum()},
            "BTTS": {"yes": 1 - P[0, :].sum() - P[:, 0].sum() + P[0, 0]},
            "OU25": {"over": P[(x + y) >= 3].sum()},
        }
        markets["BTTS"]["no"] = 1 - markets["BTTS"]["yes"]
        markets["OU25"]["under"] = 1 - markets["OU25"]["over"]
        for closing, sd in ((False, noise_sd), (True, closing_noise_sd)):
            for mkt, probs in markets.items():
                p = np.array(list(probs.values()))
                logit = np.log(p / (1 - p)) + rng.normal(0, sd, len(p))
                q = 1 / (1 + np.exp(-logit))
                q = q / q.sum() * (1 + margin)
                for sel, qi in zip(probs, q):
                    rows.append(
                        {
                            "match_id": rec.match_id,
                            "bookmaker": bookmaker,
                            "market": mkt,
                            "selection": sel,
                            "odds": float(1 / qi),
                            "timestamp": rec.date - pd.Timedelta(hours=1 if closing else 48),
                            "is_closing": closing,
                        }
                    )
    return pd.DataFrame(rows)
