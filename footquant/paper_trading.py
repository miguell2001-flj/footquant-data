"""Paper-trading toolkit: pooled probabilities, bet sheets, a ledger, settlement and CLV.

This is the production path that the EPL backtests point to:

  1. The market is the prior. The bet365 opening 1X2 line is de-vigged (Shin)
     and inverted to market-implied rates (lambda_k, mu_k).
  2. The Dixon-Coles xG model supplies (lambda_m, mu_m). Both enter a log-linear
     pool with weight w on the model (fit w on history with
     `fit_pool_weight_on_history`; ~0.2 on EPL 2016-21).
  3. The early-payout increment is simulated from the pooled rates, calibrated
     to the realised "led by two, did not win" rates (ValueConfig.early_payout_scale)
     and added to the pooled win probability; that is the probability the
     cash-flow depends on. A bet needs an expected return of `edge_margin`
     per unit staked at the price on offer (edge_metric 'ev').
  4. Kelly (quarter) is sized on the pooled probability, so stakes track the
     1-2 % real expectation instead of the model's raw disagreement with the line.

Workflow (see examples/paper_trade.py):

    sheet  -> fit on history, read upcoming fixtures + prices, write the bets to a ledger
    settle -> once results (and closing lines) are in the season file, settle open bets
    report -> flat-stake yield with standard error, CLV with standard error, drawdown

Settlement uses the full-time result; the early payout is credited only when a
goal timeline (home/away_max_lead) is available, otherwise a half-time lead of
two or more is used as a lower bound and the bet is flagged `ep_unverified`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.metrics import outcome_index
from .backtest.walkforward import walk_forward
from .config import BacktestConfig, ModelConfig, StakingConfig, ValueConfig
from .data.schema import validate_matches
from .market.blend import fit_pool_weight, pool_with_market
from .market.devig import devig
from .market.odds_feed import OddsSnapshot
from .models.dixon_coles import DixonColesModel
from .staking.kelly import kelly_fraction

LEDGER_COLUMNS = [
    "run_id", "placed_at", "match_id", "date", "league", "home", "away", "selection", "odds",
    "p_model", "p_market", "p_pool", "ep_bonus", "p_paid", "edge", "ev", "kelly_full", "stake", "bankroll_at_placement",
    "settled", "won", "pnl", "ep_unverified", "closing_odds", "clv",
]


def fit_pool_weight_on_history(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    model_cfg: ModelConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
    start_date: str | pd.Timestamp | None = None,
    league: str | None = None,
) -> tuple[float, dict]:
    """Walk-forward the model over history and fit the pool weight against the
    de-vigged pre-match 1X2 line. Returns (w, info)."""
    preds = walk_forward(matches, model_cfg, bt_cfg, start_date=start_date, league=league)
    o = odds[(odds["market"] == "1X2") & (~odds["is_closing"].astype(bool))].pivot_table(index="match_id", columns="selection", values="odds").dropna()
    preds = preds[preds["match_id"].isin(o.index)]
    o = o.loc[preds["match_id"]]
    pk = np.array([devig([h, d, a], "shin") for h, d, a in zip(o["home"], o["draw"], o["away"])])
    pm = preds[["p_home", "p_draw", "p_away"]].values
    y = outcome_index(preds["home_goals"].values, preds["away_goals"].values)
    w, info = fit_pool_weight(pm, pk, y)
    info["n"] = int(len(preds))
    return w, info


def pooled_fixture(
    model: DixonColesModel,
    snapshot: OddsSnapshot,
    home: str,
    away: str,
    w: float,
    rho_market: float | None = None,
    over25: float | None = None,
    max_goals: int = 8,
    n_sims: int = 50_000,
    seed: int = 0,
    neutral: bool = False,
    value_cfg: ValueConfig | None = None,
) -> dict:
    """Pooled 1X2 probabilities and early-payout increments for one fixture."""
    p_model = model.predict(home, away, neutral)
    if rho_market is not None:
        p_model = {**p_model, "rho": rho_market}
    pooled, ep = pool_with_market(p_model, snapshot.odds_vector, w, over25, max_goals, n_sims, seed, value_cfg=value_cfg)
    return {
        "p_model": np.array([pooled["p_model_home"], pooled["p_model_draw"], pooled["p_model_away"]]),
        "p_market": np.array([pooled["m_home"], pooled["m_draw"], pooled["m_away"]]),
        "p_pool": np.array([pooled["p_home"], pooled["p_draw"], pooled["p_away"]]),
        "lam_model": pooled["lam_model"], "mu_model": pooled["mu_model"], "lam_market": pooled["lam_market"], "mu_market": pooled["mu_market"],
        "lam_pool": pooled["lam"], "mu_pool": pooled["mu"], "bonus": np.array([ep["p_home_bonus"], 0.0, ep["p_away_bonus"]]),
    }


def make_bet_sheet(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    snapshots: list[OddsSnapshot],
    w: float,
    bankroll: float,
    model_cfg: ModelConfig | None = None,
    value_cfg: ValueConfig | None = None,
    staking_cfg: StakingConfig | None = None,
    as_of: pd.Timestamp | None = None,
    run_id: str | None = None,
    models: dict[str, DixonColesModel] | None = None,
    bookmaker_of: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Price upcoming fixtures and return the stakes to record.

    `fixtures`: match_id, date, league, home, away[, neutral]; `snapshots`: the
    pre-match 1X2 (and optionally OU25) prices per match. Only 1X2 is staked.
    `models`: prefitted model per league (e.g. one joint cross-league fit shared
    by the UEFA competitions); when given, nothing is refitted from `history`.
    """
    model_cfg = model_cfg or ModelConfig(xg_weight=0.6, xi=0.003)
    value_cfg = value_cfg or ValueConfig(edge_margin=0.02)
    staking_cfg = staking_cfg or StakingConfig()
    hist = validate_matches(history)
    as_of = pd.Timestamp(as_of) if as_of is not None else hist["date"].max()
    if models is None:
        models = {}
        for league, part in hist[hist["date"] <= as_of].groupby("league"):
            if len(part) >= 20:
                models[league] = DixonColesModel(model_cfg)
                models[league].fit(part, as_of=as_of)
    by_match: dict[str, dict[str, OddsSnapshot]] = {}
    for s in snapshots:
        by_match.setdefault(s.match_id, {})[s.market] = s
    rows = []
    placed_at = pd.Timestamp.now().isoformat(timespec="seconds")
    run_id = run_id or placed_at
    seed = value_cfg.seed
    for rec in fixtures.to_dict("records"):
        model = models.get(rec["league"])
        snaps = by_match.get(rec["match_id"], {})
        if model is None or "1X2" not in snaps:
            continue
        over25 = None
        if "OU25" in snaps:
            over25 = float(devig(snaps["OU25"].odds_vector, "shin")[0])
        f = pooled_fixture(model, snaps["1X2"], rec["home"], rec["away"], w, over25=over25, max_goals=model_cfg.max_goals,
                           n_sims=value_cfg.n_sims, seed=seed, neutral=bool(rec.get("neutral", False)), value_cfg=value_cfg)
        seed += 1
        eligible = value_cfg.early_payout and rec["league"] in value_cfg.early_payout_competitions
        best = None
        for j, sel in enumerate(("home", "draw", "away")):
            o = snaps["1X2"].prices[sel]
            bonus = float(f["bonus"][j]) * value_cfg.early_payout_multiplier(float(f["p_market"][j])) if eligible else 0.0
            p_paid = min(float(f["p_pool"][j]) + bonus, 1.0)
            edge = p_paid - float(f["p_market"][j])
            ev = p_paid * o - 1.0
            clears = ev >= value_cfg.edge_margin if value_cfg.edge_metric == "ev" else edge >= value_cfg.edge_margin
            if clears and ev > 0 and (best is None or ev > best["ev"]):
                k = kelly_fraction(p_paid, o)
                stake = min(k * staking_cfg.kelly_fraction, staking_cfg.max_stake_frac) * bankroll
                if staking_cfg.round_to > 0:
                    stake = float(np.floor(stake / staking_cfg.round_to) * staking_cfg.round_to)
                if stake <= 0 or stake < staking_cfg.min_stake:
                    continue
                best = {
                    "run_id": run_id, "placed_at": placed_at, "match_id": rec["match_id"], "date": pd.Timestamp(rec["date"]),
                    "league": rec["league"], "home": rec["home"], "away": rec["away"], "selection": sel, "odds": float(o),
                    "p_model": float(f["p_model"][j]), "p_market": float(f["p_market"][j]), "p_pool": float(f["p_pool"][j]),
                    "ep_bonus": bonus, "p_paid": p_paid, "edge": edge, "ev": ev, "kelly_full": k, "stake": stake,
                    "bankroll_at_placement": bankroll, "settled": False, "won": np.nan, "pnl": np.nan,
                    "ep_unverified": False, "closing_odds": np.nan, "clv": np.nan,
                }
        if best is not None:
            rows.append(best)
    sheet = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    total = sheet["stake"].sum()
    if total > staking_cfg.max_total_frac * bankroll and total > 0:  # exposure cap across the sheet
        sheet["stake"] = np.floor(sheet["stake"] * staking_cfg.max_total_frac * bankroll / total / max(staking_cfg.round_to, 1e-9)) * max(staking_cfg.round_to, 1e-9)
    return sheet


class Ledger:
    """CSV-backed record of paper (or real) bets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
            for c in LEDGER_COLUMNS:
                if c not in self.df.columns:
                    self.df[c] = np.nan
            self._coerce()
        else:
            self.df = pd.DataFrame(columns=LEDGER_COLUMNS)

    _NUMERIC = ["odds", "p_model", "p_market", "p_pool", "ep_bonus", "p_paid", "edge", "ev", "kelly_full", "stake",
                "bankroll_at_placement", "won", "pnl", "closing_odds", "clv"]

    def _coerce(self) -> None:
        for c in self._NUMERIC:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce").astype(float)
        for c in ("settled", "ep_unverified"):
            self.df[c] = self.df[c].map(lambda v: str(v).lower() == "true" if isinstance(v, str) else bool(v) if pd.notna(v) else False)
        self.df["date"] = pd.to_datetime(self.df["date"])

    def append(self, sheet: pd.DataFrame) -> int:
        new = sheet[~sheet["match_id"].isin(self.df["match_id"])] if len(self.df) else sheet
        self.df = pd.concat([self.df, new], ignore_index=True) if len(self.df) else new.copy()
        self._coerce()
        self.save()
        return int(len(new))

    def save(self) -> None:
        self.df.to_csv(self.path, index=False)

    def settle(self, results: pd.DataFrame, closing_odds: pd.DataFrame | None = None, lead: int = 2) -> int:
        """Settle open bets against a match table (match_id or (date, home, away) join).

        `results` may carry home_max_lead/away_max_lead (goal timeline) or
        ht_home_goals/ht_away_goals (half-time score) for the early payout.
        `closing_odds` is an ODDS_COLUMNS frame with is_closing=True rows.
        """
        if len(self.df) == 0:
            return 0
        res = results.copy()
        res["date"] = pd.to_datetime(res["date"]).dt.normalize()
        key = res.set_index(["date", "home", "away"])
        closing = None
        if closing_odds is not None and len(closing_odds):
            closing = closing_odds[(closing_odds["market"] == "1X2") & (closing_odds["is_closing"].astype(bool))].pivot_table(
                index="match_id", columns="selection", values="odds")
        n = 0
        settled = self.df["settled"].astype(bool)
        unverified = self.df["ep_unverified"].astype(bool) if "ep_unverified" in self.df.columns else pd.Series(False, index=self.df.index)
        for i, b in self.df[(~settled) | (settled & unverified)].iterrows():
            recheck = bool(settled.loc[i])  # settled on a half-time proxy: redo it only if the goal timeline is now known
            k = (pd.Timestamp(b["date"]).normalize(), b["home"], b["away"])
            r = res[res["match_id"] == b["match_id"]] if "match_id" in res.columns and (res["match_id"] == b["match_id"]).any() else None
            if r is None or len(r) == 0:
                if k not in key.index:
                    continue
                r = key.loc[[k]].reset_index()
            r = r.iloc[0]
            if pd.isna(r.get("home_goals")) or pd.isna(r.get("away_goals")):
                continue
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
            sel = b["selection"]
            if recheck:
                ml_known = r.get(f"{sel}_max_lead")
                if ml_known is None or pd.isna(ml_known):
                    continue
            won = {"home": hg > ag, "draw": hg == ag, "away": ag > hg}[sel]
            unverified = False
            if sel in ("home", "away") and b["ep_bonus"] > 0 and not won:
                own_goals = hg if sel == "home" else ag
                ml = r.get(f"{sel}_max_lead")
                if ml is not None and not pd.isna(ml):
                    won = int(ml) >= lead
                elif own_goals < lead:
                    won = False  # could never have led by `lead`
                else:
                    hth, hta = r.get("ht_home_goals"), r.get("ht_away_goals")
                    if hth is not None and not pd.isna(hth):
                        ht_lead = (int(hth) - int(hta)) if sel == "home" else (int(hta) - int(hth))
                        won = ht_lead >= lead
                        unverified = not won  # a 2-goal lead may have happened outside half-time
                    else:
                        unverified = True
            pnl = b["stake"] * (b["odds"] - 1.0) if won else -b["stake"]
            self.df.loc[i, "settled"] = True
            self.df.loc[i, "won"] = 1.0 if won else 0.0  # stored as float: 1 won, 0 lost, NaN open
            self.df.loc[i, "pnl"] = float(pnl)
            self.df.loc[i, "ep_unverified"] = bool(unverified)
            if closing is not None and b["match_id"] in closing.index:
                row = closing.loc[b["match_id"]]
                if row.notna().all():
                    fair = dict(zip(["home", "draw", "away"], devig([row["home"], row["draw"], row["away"]], "shin")))
                    self.df.loc[i, "closing_odds"] = float(row[sel])
                    self.df.loc[i, "clv"] = float(b["odds"] * fair[sel] - 1.0)
            n += 1
        self._coerce()
        self.save()
        return n

    def report(self) -> dict:
        d = self.df[self.df["settled"].astype(bool)].copy()
        out = {"n_open": int((~self.df["settled"].astype(bool)).sum()), "n_settled": int(len(d))}
        if len(d) == 0:
            return out
        won = d["won"] == 1.0
        flat = np.where(won, d["odds"] - 1.0, -1.0)
        out.update(
            {
                "staked": float(d["stake"].sum()),
                "pnl": float(d["pnl"].sum()),
                "yield_on_turnover": float(d["pnl"].sum() / d["stake"].sum()) if d["stake"].sum() > 0 else np.nan,
                "flat_yield": float(flat.mean()),
                "flat_yield_se": float(flat.std(ddof=1) / np.sqrt(len(flat))) if len(flat) > 1 else np.nan,
                "hit_rate": float(won.mean()),
                "avg_odds": float(d["odds"].mean()),
                "avg_ev_at_placement": float(d["ev"].mean()),
                "n_clv": int(d["clv"].notna().sum()),
                "clv_mean": float(d["clv"].mean()) if d["clv"].notna().any() else np.nan,
                "clv_se": float(d["clv"].std(ddof=1) / np.sqrt(d["clv"].notna().sum())) if d["clv"].notna().sum() > 1 else np.nan,
                "ep_unverified": int(d["ep_unverified"].astype(bool).sum()),
            }
        )
        path = (d.sort_values("date")["pnl"].cumsum() + d["bankroll_at_placement"].iloc[0]).values
        peak = np.maximum.accumulate(path)
        out["max_drawdown"] = float(np.max((peak - path) / peak)) if len(path) else 0.0
        return out
