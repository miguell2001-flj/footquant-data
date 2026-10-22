"""End-to-end orchestration: fit -> predict -> value -> stake.

Typical live use:

    pipe = BettingPipeline()
    pipe.fit(history)                        # one DC model per league + Elo replay
    probs = pipe.predict(fixtures)           # 1X2 / BTTS / O-U 2.5 + early-payout increments
    bets  = pipe.find_value(probs, snapshots)  # vs bet365 prices
    plan  = pipe.recommend_stakes(bets, bankroll=2500)

`history` is the validated match table (all competitions), `fixtures` the
upcoming matches (match_id, date, league, home, away[, neutral]), `snapshots`
an `OddsSnapshot` list from `market.odds_feed`.

The contextual boosting layer is optional: train it with `fit_context_layer`
on walk-forward output (so the base rates are out-of-sample) and the pipeline
applies it to the leagues named in `context_leagues`.
"""

from __future__ import annotations

import pandas as pd

from .config import EloConfig, ModelConfig, StakingConfig, ValueConfig
from .data.elo import LeagueElo
from .data.features import build_feature_frame
from .data.schema import validate_matches
from .market.blend import pool_with_market
from .market.odds_feed import OddsSnapshot
from .market.value import ValueBet, evaluate_snapshot, value_table
from .models.context_boost import ContextualRateAdjuster, team_rows_from_predictions
from .models.dixon_coles import DixonColesModel
from .models.early_payout import simulate_early_payout
from .models.score_matrix import market_probs, score_matrix
from .staking.kelly import allocate_stakes


def team_league_map(domestic: pd.DataFrame) -> dict[str, str]:
    """Each team's domestic league: the league it has played most matches in."""
    long = pd.concat([domestic[["home", "league"]].rename(columns={"home": "team"}), domestic[["away", "league"]].rename(columns={"away": "team"})])
    counts = long.groupby(["team", "league"]).size().reset_index(name="n").sort_values(["team", "n"], ascending=[True, False])
    return dict(counts.drop_duplicates("team")[["team", "league"]].values)


class BettingPipeline:
    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        elo_cfg: EloConfig | None = None,
        value_cfg: ValueConfig | None = None,
        staking_cfg: StakingConfig | None = None,
        context_leagues: frozenset[str] = frozenset(),
        context_shrink: float = 0.7,
        pool_weight: float | None = None,
        joint_competitions: frozenset[str] = frozenset(),
    ) -> None:
        self.model_cfg = model_cfg or ModelConfig()
        self.elo_cfg = elo_cfg or EloConfig()
        self.value_cfg = value_cfg or ValueConfig()
        self.staking_cfg = staking_cfg or StakingConfig()
        self.context_leagues = context_leagues
        self.context_shrink = context_shrink
        self.pool_weight = pool_weight
        """None -> the framework's pure-model comparison. A float in [0, 1] ->
        the market-as-prior path: every market is priced from a log-linear pool
        of model and de-vigged 1X2 line at this weight on the model (fit it with
        `paper_trading.fit_pool_weight_on_history`; ~0.2 on EPL 2016-21)."""
        self.joint_competitions = joint_competitions
        """Cross-league competitions (Champions League, Europa League...) priced
        by one joint Dixon-Coles fit with league-strength offsets; teams are
        mapped to the domestic league they appear in. Domestic leagues keep
        their own separate fits."""
        self.joint_model: DixonColesModel | None = None
        self.team_league: dict[str, str] = {}
        self.models: dict[str, DixonColesModel] = {}
        self.elo = LeagueElo(self.elo_cfg)
        self.adjuster: ContextualRateAdjuster | None = None
        self.history: pd.DataFrame | None = None
        self.stadiums: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp | None = None, stadiums: pd.DataFrame | None = None) -> "BettingPipeline":
        df = validate_matches(history)
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
        df = df[df["date"] <= as_of]
        self.history = df
        self.stadiums = stadiums
        self.elo = LeagueElo(self.elo_cfg)
        self.elo.run(df)
        self.models = {}
        domestic = df[~df["league"].isin(self.joint_competitions)]
        for league, part in domestic.groupby("league"):
            if len(part) < 20:
                continue
            m = DixonColesModel(self.model_cfg)
            m.fit(part, as_of=as_of)
            self.models[league] = m
        if self.joint_competitions:
            self.team_league = team_league_map(domestic)
            known = df["home"].isin(self.team_league) & df["away"].isin(self.team_league)
            joint_df = df[known]
            if len(joint_df) >= 20 and joint_df["league"].isin(self.joint_competitions).any():
                self.joint_model = DixonColesModel(self.model_cfg)
                self.joint_model.fit(joint_df, as_of=as_of, team_league=self.team_league)
                for comp in self.joint_competitions:
                    self.models[comp] = self.joint_model
        return self

    def fit_context_layer(self, walkforward_preds: pd.DataFrame, feats: pd.DataFrame, target: str = "y_goals", **kw) -> "BettingPipeline":
        """Train the boosted rate adjuster on out-of-sample base rates."""
        rows = team_rows_from_predictions(walkforward_preds, feats)
        self.adjuster = ContextualRateAdjuster(shrink=self.context_shrink, **kw).fit(rows, target=target)
        return self

    # --------------------------------------------------------------- predicting
    def predict(self, fixtures: pd.DataFrame, absences: pd.DataFrame | None = None) -> pd.DataFrame:
        """Model probabilities for upcoming fixtures, including early-payout increments."""
        fx = fixtures.copy()
        fx["date"] = pd.to_datetime(fx["date"])
        rows = []
        for rec in fx.to_dict("records"):
            model = self.models.get(rec["league"])
            if model is None:
                continue
            neutral = bool(rec.get("neutral", False))
            lam, mu = model.predict_rates(rec["home"], rec["away"], neutral)
            rows.append({"match_id": rec["match_id"], "date": rec["date"], "league": rec["league"], "home": rec["home"],
                         "away": rec["away"], "neutral": neutral, "lam": lam, "mu": mu, "rho": model.params.rho})
        preds = pd.DataFrame(rows)
        if preds.empty:
            return preds
        if self.adjuster is not None and self.history is not None:
            mask = preds["league"].isin(self.context_leagues)
            if mask.any():
                # congestion/rest features need the fixtures appended to history
                combined = pd.concat([self.history, fx.reindex(columns=self.history.columns)], ignore_index=True)
                combined = combined.sort_values("date").reset_index(drop=True)
                elo_pre = self.elo.predict_fixtures(fx)
                feats = build_feature_frame(combined, self.stadiums, absences, elo_pre)
                feats = feats[feats["match_id"].isin(preds.loc[mask, "match_id"])]
                adj = self.adjuster.adjust(preds[mask], feats)
                preds.loc[mask, "lam"] = adj["lam_adj"].values
                preds.loc[mask, "mu"] = adj["mu_adj"].values
                preds.loc[mask, "home_factor"] = adj["home_factor"].values
                preds.loc[mask, "away_factor"] = adj["away_factor"].values
        out = []
        seed = self.value_cfg.seed
        for rec in preds.to_dict("records"):
            p = market_probs(score_matrix(rec["lam"], rec["mu"], rec["rho"], self.model_cfg.max_goals))
            rec.update(p)
            if self.value_cfg.early_payout and rec["league"] in self.value_cfg.early_payout_competitions:
                ep = simulate_early_payout(rec["lam"], rec["mu"], self.value_cfg, seed=seed)
                seed += 1
                rec.update({k: v for k, v in ep.items() if k.endswith("_bonus") or k.endswith("_reached")})
            out.append(rec)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------- value
    def find_value(self, probs: pd.DataFrame, snapshots: list[OddsSnapshot]) -> list[ValueBet]:
        by_id = {r["match_id"]: r for r in probs.to_dict("records")}
        by_match: dict[str, dict[str, OddsSnapshot]] = {}
        for s in snapshots:
            by_match.setdefault(s.match_id, {})[s.market] = s
        bets: list[ValueBet] = []
        seed = self.value_cfg.seed + 10_000
        for mid, snaps in by_match.items():
            rec = by_id.get(mid)
            if rec is None:
                continue
            ep = None
            if "p_home_bonus" in rec and pd.notna(rec.get("p_home_bonus")):
                ep = {"p_home_bonus": rec["p_home_bonus"], "p_away_bonus": rec["p_away_bonus"]}
            if self.pool_weight is not None and "1X2" in snaps:
                over25 = None
                if "OU25" in snaps:
                    from .market.devig import devig

                    over25 = float(devig(snaps["OU25"].odds_vector, self.value_cfg.devig_method)[0])
                rec, ep = pool_with_market(rec, snaps["1X2"].odds_vector, self.pool_weight, over25, self.model_cfg.max_goals,
                                           self.value_cfg.n_sims, seed, self.value_cfg.devig_method, value_cfg=self.value_cfg)
                seed += 1
            for s in snaps.values():
                bets.extend(evaluate_snapshot(rec, s, self.value_cfg, rec["league"], ep))
        return bets

    def value_report(self, bets: list[ValueBet], flagged_only: bool = True) -> pd.DataFrame:
        tbl = value_table(bets)
        if flagged_only and len(tbl):
            tbl = tbl[tbl["flagged"]]
        return tbl.sort_values("edge", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------------- staking
    def recommend_stakes(self, bets: list[ValueBet], bankroll: float) -> pd.DataFrame:
        return allocate_stakes(bets, bankroll, self.staking_cfg)
