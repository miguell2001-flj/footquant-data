"""Walk-forward (rolling out-of-sample) validation and betting simulation (Phase 6).

Look-ahead rules enforced here:
  * At window start t the model is re-fitted on matches dated strictly before t,
    with time weights measured from t.
  * Every fixture in [t, t + refit_every_days) is predicted from that fit only.
  * Elo and contextual features are built from prior fixtures only (see data/).
  * Bets are placed at the odds available before kick-off (is_closing=False);
    closing lines are used only to measure closing-line value.
  * Early-payout settlement needs the realised goal timeline (home_max_lead /
    away_max_lead). Without those columns the simulation disables the
    early-payout adjustment and says so, rather than assume payouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BacktestConfig, ModelConfig, StakingConfig, ValueConfig
from ..data.schema import validate_matches
from ..market.devig import devig
from ..market.odds_feed import snapshots_from_frame
from ..market.value import evaluate_snapshot
from ..models.dixon_coles import DixonColesModel
from ..models.early_payout import simulate_early_payout
from ..models.score_matrix import market_probs, score_matrix
from ..staking.kelly import allocate_stakes
from .metrics import closing_line_value, evaluate_predictions, max_drawdown


def walk_forward(
    matches: pd.DataFrame,
    model_cfg: ModelConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    league: str | None = None,
    team_league: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Out-of-sample predictions for every match from `start_date` on.

    Returns one row per predicted match with lam, mu, rho, p_home/p_draw/p_away,
    p_btts_yes, p_over25, the realised goals (for scoring) and the fit date.
    With `team_league` the frame may span several leagues plus cross-league
    competitions and every refit is the joint fit with league offsets.
    """
    model_cfg = model_cfg or ModelConfig()
    bt_cfg = bt_cfg or BacktestConfig()
    df = validate_matches(matches)
    if league is not None:
        df = df[df["league"] == league].reset_index(drop=True)
    if team_league is not None:
        df = df[df["home"].isin(team_league) & df["away"].isin(team_league)].reset_index(drop=True)
    elif df["league"].nunique() != 1:
        raise ValueError("walk_forward fits one league at a time; pass `league=` (or `team_league=` for a joint fit)")
    if start_date is None:
        if len(df) <= bt_cfg.min_train_matches:
            raise ValueError("not enough matches for the requested min_train_matches")
        start_date = df["date"].iloc[bt_cfg.min_train_matches]
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date) if end_date is not None else df["date"].max()
    step = pd.Timedelta(days=bt_cfg.refit_every_days)

    model = DixonColesModel(model_cfg)
    rows = []
    t = start_date
    while t <= end_date:
        test = df[(df["date"] >= t) & (df["date"] < t + step)]
        if len(test):
            train = df[df["date"] < t]
            if len(train) < 20:
                t += step
                continue
            params = model.fit(train, as_of=t, team_league=team_league)
            for rec in test.itertuples(index=False):
                neutral = bool(getattr(rec, "neutral", False))
                lam, mu = params.rates(rec.home, rec.away, neutral)
                p = market_probs(score_matrix(lam, mu, params.rho, model_cfg.max_goals))
                rows.append(
                    {
                        "match_id": rec.match_id,
                        "date": rec.date,
                        "league": rec.league,
                        "home": rec.home,
                        "away": rec.away,
                        "fit_date": t,
                        "lam": lam,
                        "mu": mu,
                        "rho": params.rho,
                        **p,
                        "home_goals": rec.home_goals,
                        "away_goals": rec.away_goals,
                        "home_xg": rec.home_xg,
                        "away_xg": rec.away_xg,
                        "home_known": rec.home in params.index,
                        "away_known": rec.away in params.index,
                    }
                )
        t += step
    return pd.DataFrame(rows)


def tune_xi(
    matches: pd.DataFrame,
    model_cfg: ModelConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
    xi_grid: tuple[float, ...] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Grid-search the time-decay rate on out-of-sample RPS (Brier and log loss
    reported alongside). Returns one row per xi, sorted by RPS."""
    from dataclasses import replace

    model_cfg = model_cfg or ModelConfig()
    bt_cfg = bt_cfg or BacktestConfig()
    grid = xi_grid or bt_cfg.xi_grid
    rows = []
    for xi in grid:
        preds = walk_forward(matches, replace(model_cfg, xi=xi), bt_cfg, **kwargs)
        m = evaluate_predictions(preds)
        rows.append({"xi": xi, "half_life_days": np.inf if xi == 0 else np.log(2) / xi, **m})
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)


def _settle(selection: str, market: str, hg: int, ag: int, max_lead_home, max_lead_away, lead: int, eligible: bool) -> bool:
    if market == "1X2":
        if selection == "home":
            won = hg > ag
            if eligible and max_lead_home is not None and not np.isnan(max_lead_home):
                won = won or max_lead_home >= lead
            return bool(won)
        if selection == "away":
            won = ag > hg
            if eligible and max_lead_away is not None and not np.isnan(max_lead_away):
                won = won or max_lead_away >= lead
            return bool(won)
        return hg == ag
    if market == "BTTS":
        return (hg > 0 and ag > 0) == (selection == "yes")
    if market == "OU25":
        return (hg + ag >= 3) == (selection == "over")
    raise ValueError(market)


def simulate_betting(
    preds: pd.DataFrame,
    odds: pd.DataFrame,
    value_cfg: ValueConfig | None = None,
    staking_cfg: StakingConfig | None = None,
    bankroll: float = 1000.0,
    bookmaker: str = "bet365",
    max_lead: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Bet the walk-forward predictions at pre-match odds, bet by bet, compounding.

    `odds` is an ODDS_COLUMNS frame; `max_lead` (match_id, home_max_lead,
    away_max_lead) enables early-payout settlement. Returns (bets, summary).
    """
    value_cfg = value_cfg or ValueConfig()
    staking_cfg = staking_cfg or StakingConfig()
    pre = snapshots_from_frame(odds[~odds["is_closing"].astype(bool)], bookmaker)
    closing = {
        (s.match_id, s.market): s for s in snapshots_from_frame(odds[odds["is_closing"].astype(bool)], bookmaker)
    }
    by_match: dict[str, list] = {}
    for s in pre:
        by_match.setdefault(s.match_id, []).append(s)

    lead_lookup = {}
    if max_lead is not None:
        lead_lookup = {r.match_id: (r.home_max_lead, r.away_max_lead) for r in max_lead.itertuples(index=False)}
    elif {"home_max_lead", "away_max_lead"} <= set(preds.columns):
        lead_lookup = {r.match_id: (r.home_max_lead, r.away_max_lead) for r in preds.itertuples(index=False)}
    early_payout_enabled = value_cfg.early_payout and bool(lead_lookup)
    if value_cfg.early_payout and not lead_lookup:
        print("simulate_betting: no goal-timeline columns -> early-payout adjustment disabled for this run")
    from dataclasses import replace

    vcfg = value_cfg if early_payout_enabled else replace(value_cfg, early_payout=False)

    rng_seed = vcfg.seed
    bank = bankroll
    path = [bank]
    bets = []
    preds = preds.sort_values(["date", "match_id"]).reset_index(drop=True)
    # bets are settled per matchday: stakes within one day are sized from the same bankroll
    for day, day_preds in preds.groupby(preds["date"].dt.normalize(), sort=True):
        candidates = []
        for rec in day_preds.to_dict("records"):
            snaps = by_match.get(rec["match_id"], [])
            if not snaps:
                continue
            ep = None
            if vcfg.early_payout and rec["league"] in vcfg.early_payout_competitions:
                ep = simulate_early_payout(rec["lam"], rec["mu"], vcfg, seed=rng_seed, n_sims=min(vcfg.n_sims, 50_000))
                rng_seed += 1
            for s in snaps:
                candidates.extend(evaluate_snapshot(rec, s, vcfg, rec["league"], ep))
        stakes = allocate_stakes(candidates, bank, staking_cfg)
        pnl_day = 0.0
        for st in stakes.itertuples(index=False):
            rec = day_preds[day_preds["match_id"] == st.match_id].iloc[0]
            ml = lead_lookup.get(st.match_id, (np.nan, np.nan))
            eligible = early_payout_enabled and rec["league"] in vcfg.early_payout_competitions
            won = _settle(st.selection, st.market, int(rec["home_goals"]), int(rec["away_goals"]), ml[0], ml[1], vcfg.early_payout_lead, eligible)
            pnl = st.stake * (st.odds - 1.0) if won else -st.stake
            pnl_day += pnl
            clv = np.nan
            cs = closing.get((st.match_id, st.market))
            if cs is not None:
                fair_close = dict(zip(cs.selections, devig(cs.odds_vector, vcfg.devig_method)))
                clv = closing_line_value(st.odds, fair_close[st.selection])
            bets.append(
                {
                    "date": rec["date"],
                    "match_id": st.match_id,
                    "home": rec["home"],
                    "away": rec["away"],
                    "market": st.market,
                    "selection": st.selection,
                    "odds": st.odds,
                    "p_model_paid": st.p_model_paid,
                    "edge": st.edge,
                    "ev": st.ev,
                    "stake": st.stake,
                    "stake_frac": st.stake_frac,
                    "won": won,
                    "pnl": pnl,
                    "clv": clv,
                    "bankroll_before": bank,
                }
            )
        bank += pnl_day
        path.append(bank)
    bets_df = pd.DataFrame(bets)
    staked = float(bets_df["stake"].sum()) if len(bets_df) else 0.0
    summary = {
        "n_bets": int(len(bets_df)),
        "final_bankroll": bank,
        "return_on_bankroll": bank / bankroll - 1.0,
        "yield_on_turnover": (float(bets_df["pnl"].sum()) / staked) if staked > 0 else 0.0,
        "hit_rate": float(bets_df["won"].mean()) if len(bets_df) else np.nan,
        "avg_odds": float(bets_df["odds"].mean()) if len(bets_df) else np.nan,
        "avg_edge": float(bets_df["edge"].mean()) if len(bets_df) else np.nan,
        "mean_clv": float(bets_df["clv"].mean()) if len(bets_df) and bets_df["clv"].notna().any() else np.nan,
        "max_drawdown": max_drawdown(path),
        "early_payout_enabled": early_payout_enabled,
    }
    return bets_df, summary
