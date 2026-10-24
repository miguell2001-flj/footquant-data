"""Betfair Exchange as the fair-price reference, and the bet365 early-payout curve.

Why: the walk-forward pool fit gives the Dixon-Coles model a weight of 0.00-0.01 against the market, so the
model cannot find value by itself. What does work (backtest on 3,700 big-5 matches, 2024/25 to 2026/27):

* bet365's price on its own almost never beats the exchange fair price (bet365 margin ~5.5%, exchange ~0.9%);
* but bet365 also pays a home/away 1X2 bet as a winner once the team leads by two goals. That happens on ~2% of
  bets whatever the price, so it is worth ~2% x odds. Priced off the exchange, bet365 + early payout clears a
  2% value bar on ~60 bets a season at odds <= 2.5 (closing-line value +2.1% vs the exchange close) and on
  ~370 bets a season at odds <= 4.0 (+2.7%).

Everything here works from football-data.co.uk columns: BFEH/BFED/BFEA (exchange, at the time the file was
collected), BFECH/BFECD/BFECA (exchange close), B365H/D/A, and the named bookmakers in fixtures.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.ingest import _canon, read_football_data_csv

SELS = ("home", "draw", "away")
EX_PRE = ("BFEH", "BFED", "BFEA")
EX_CLOSE = ("BFECH", "BFECD", "BFECA")
# named bookmakers that fixtures.csv carries (column prefix -> display name), for the line-shopping flags
BOOKS = {"B365": "bet365", "BW": "bwin", "BV": "BetVictor", "BFD": "Betfred", "PP": "Paddy Power", "SKB": "Sky Bet",
         "WH": "William Hill", "1XB": "1xBet", "BMGM": "BetMGM", "CL": "Coolbet", "LB": "Ladbrokes", "Max": "best price"}


def power_devig(odds) -> np.ndarray:
    """Fair probabilities from a price vector (or an n x k array) by the power method: p_i = (1/o_i)^k, sum = 1.
    Less longshot bias than proportional normalisation; on exchange prices the two barely differ."""
    o = np.atleast_2d(np.asarray(odds, dtype=float))
    q = 1.0 / o
    k = np.ones(len(q))
    for _ in range(60):
        qk = q ** k[:, None]
        s, ds = qk.sum(1), (np.log(q) * qk).sum(1)
        k = k - (s - 1.0) / ds
    p = q ** k[:, None]
    p = p / p.sum(1, keepdims=True)
    return p[0] if np.ndim(odds) == 1 else p


def exchange_table(path: str | Path, league, season: str, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per match: match_id (same convention as ingest), exchange fair probabilities before kick-off
    (fx_*) and at the close (fc_*), plus every named bookmaker's 1X2 price found in the file."""
    raw = read_football_data_csv(path)
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    if isinstance(league, dict):  # fixtures.csv: league from the Div column
        raw = raw[raw["Div"].isin(league)]
        lg = raw["Div"].map(league)
    else:
        lg = pd.Series(league, index=raw.index)
    date = pd.to_datetime(raw["Date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    if "Time" in raw.columns:
        date = date + pd.to_timedelta(raw["Time"].fillna("15:00").astype(str) + ":00", errors="coerce").fillna(pd.Timedelta(hours=15))
    home = raw["HomeTeam"].map(lambda t: _canon(t, aliases))
    away = raw["AwayTeam"].map(lambda t: _canon(t, aliases))
    out = pd.DataFrame({"match_id": [f"{l}|{season}|{d.date()}|{h}|{a}" for l, d, h, a in zip(lg, date, home, away)],
                        "league": lg.values, "date": date.values})
    for cols, pre in ((EX_PRE, "fx"), (EX_CLOSE, "fc")):
        if all(c in raw.columns for c in cols):
            o = raw[list(cols)].apply(pd.to_numeric, errors="coerce")
            ok = (o > 1.0).all(axis=1).values
            p = np.full((len(raw), 3), np.nan)
            if ok.any():
                p[ok] = power_devig(o.values[ok])
            for j, s in enumerate(SELS):
                out[f"{pre}_{s}"] = p[:, j]
                out[f"{pre}_odds_{s}"] = o.iloc[:, j].where(ok).values
    for b in BOOKS:
        if all(f"{b}{x}" in raw.columns for x in "HDA"):
            for x, s in zip("HDA", SELS):
                out[f"{b}_{s}"] = pd.to_numeric(raw[f"{b}{x}"], errors="coerce").values
    return out.reset_index(drop=True)


Z_CAP = 2.0  # the opponent-comeback term is capped at two standard deviations either way


@dataclass
class EpCurve:
    """P(team led by two at some point and did not win) as a smooth function of its fair win probability and of the
    OPPONENT's record of coming back from two goals down: logistic in (1, logit p, logit p^2, z), where z is the
    opponent's shrunk comeback rate standardised and capped at +-2. Fitted on matches with a known goal timeline.
    Goal-minute study, 24 Oct 2026 (5,699 sides, 120 cases): odds ratio 1.27 per standard deviation of the
    opponent's record (z = 3.0), positive in all three seasons; the team's own timing showed no effect."""
    coef: np.ndarray
    n: int
    z_mu: float = 0.0
    z_sd: float = 1.0

    @staticmethod
    def _x(p, z=None):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        lg = np.log(p / (1 - p))
        z = np.zeros_like(lg) if z is None else np.broadcast_to(np.asarray(z, dtype=float), lg.shape)
        return np.stack([np.ones_like(lg), lg, lg * lg, np.clip(np.nan_to_num(z), -Z_CAP, Z_CAP)], axis=-1)

    def zscore(self, rate):
        """Opponent's shrunk comeback rate -> the capped z the curve uses (unknown team -> 0)."""
        if rate is None or not np.isfinite(rate):
            return 0.0
        return float(np.clip((rate - self.z_mu) / self.z_sd, -Z_CAP, Z_CAP))

    @classmethod
    def fit(cls, p, ep_event, opp_rate=None) -> "EpCurve":
        p, y = np.asarray(p, float), np.asarray(ep_event, float)
        r = np.full_like(p, np.nan) if opp_rate is None else np.asarray(opp_rate, float)
        ok = np.isfinite(p) & np.isfinite(y)
        mu = float(np.nanmean(r[ok])) if np.isfinite(r[ok]).any() else 0.0
        sd = float(np.nanstd(r[ok])) if np.isfinite(r[ok]).sum() > 1 else 1.0
        z = np.where(np.isfinite(r), (r - mu) / (sd or 1.0), 0.0)
        a, y = cls._x(p[ok], z[ok]), y[ok]

        def nll(b):
            zz = a @ b
            return -np.sum(y * zz - np.logaddexp(0, zz)) + 5e-4 * np.sum(b[1:] ** 2)

        return cls(minimize(nll, np.array([-4.0, 0.0, 0.0, 0.0]), method="BFGS").x, int(ok.sum()), mu, sd or 1.0)

    def __call__(self, p, z=0.0):
        return 1.0 / (1.0 + np.exp(-(self._x(p, z) @ self.coef)))

    def describe(self) -> str:
        base = ", ".join(f"{int(q * 100)}% -> {float(self(q)):.1%}" for q in (0.2, 0.4, 0.6, 0.8))
        return f"{base}; against an opponent with a strong comeback record (+1 sd) x{float(np.exp(self.coef[3])):.2f}"

    def as_dict(self) -> dict:
        return {"coef": [round(float(c), 6) for c in self.coef], "z_mu": round(self.z_mu, 6), "z_sd": round(self.z_sd, 6), "z_cap": Z_CAP}


def comeback_rates(hist: pd.DataFrame, lead: int = 2) -> tuple[dict, dict]:
    """Each team's record of coming back from `lead` goals down (trailed by two at some point and did not lose),
    shrunk toward 10%: (came back + 1) / (trailed by two + 10).
    Returns (prior, now): prior[(match_id, team)] = the rate from that team's matches strictly before this one (for
    fitting without look-ahead); now[team] = the rate from all its matches (for pricing upcoming fixtures)."""
    h = hist[hist["home_max_lead"].notna()].sort_values("date")
    tally, prior = {}, {}
    for r in h.itertuples(index=False):
        for team, opp_lead, lost in ((r.home, r.away_max_lead, r.away_goals > r.home_goals), (r.away, r.home_max_lead, r.home_goals > r.away_goals)):
            c, t = tally.get(team, (0, 0))
            prior[(r.match_id, team)] = (c + 1) / (t + 10)
        for team, opp_lead, lost in ((r.home, r.away_max_lead, r.away_goals > r.home_goals), (r.away, r.home_max_lead, r.home_goals > r.away_goals)):
            c, t = tally.get(team, (0, 0))
            if opp_lead >= lead:
                tally[team] = (c + (0 if lost else 1), t + 1)
            else:
                tally[team] = (c, t)
    now = {team: (c + 1) / (t + 10) for team, (c, t) in tally.items()}
    return prior, now


def ep_events(hist: pd.DataFrame, ex: pd.DataFrame, lead: int = 2) -> pd.DataFrame:
    """Home and away sides with a goal timeline and an exchange price: fair p before kick-off, whether the side
    reached a `lead`-goal lead and did not win (the case only the early payout pays), and the opponent's comeback
    record from its earlier matches."""
    prior, _ = comeback_rates(hist, lead)
    m = hist.merge(ex[["match_id", "fx_home", "fx_away"]], on="match_id", how="inner")
    m = m[m["home_max_lead"].notna() & m["fx_home"].notna()]
    rows = []
    for s, o in (("home", "away"), ("away", "home")):
        won = m[f"{s}_goals"] > m[f"{o}_goals"]
        rows.append(pd.DataFrame({"p": m[f"fx_{s}"].values, "ep": ((m[f"{s}_max_lead"] >= lead) & ~won).astype(float).values,
                                  "opp_rate": [prior.get((mid, t), np.nan) for mid, t in zip(m["match_id"], m[o])]}))
    return pd.concat(rows, ignore_index=True)


def screen(fixtures: pd.DataFrame, ex: pd.DataFrame, curve: EpCurve, margin: float, bankroll: float, kelly_mult: float,
           max_odds: float | None, ep_leagues, run_id: str, placed_at: str, max_stake_frac: float = 0.05,
           round_to: float = 0.5, w_model: dict | None = None, p_model: dict | None = None, opp_rate: dict | None = None):
    """Price every fixture off the exchange.

    Returns (bets, card): `bets` are LEDGER rows for the bet365 selections that clear `margin` (one per match, the
    best by value, odds <= max_odds); `card` has one row per fixture with the fair price and the minimum price to
    take at bet365 (early payout included) and at any other bookmaker, plus named bookmakers already paying it.
    """
    from ..paper_trading import LEDGER_COLUMNS
    from ..staking.kelly import kelly_fraction

    fx = fixtures.merge(ex.drop(columns=["league", "date"], errors="ignore"), on="match_id", how="left")
    bets, card = [], []
    for r in fx.to_dict("records"):
        p = np.array([r.get(f"fx_{s}", np.nan) for s in SELS], dtype=float)
        if not np.isfinite(p).all():
            card.append({"match_id": r["match_id"], "date": r["date"], "league": r["league"], "home": r["home"], "away": r["away"],
                         "priced": False, "note": "no exchange price in the data yet"})
            continue
        pm = p_model.get(r["match_id"]) if p_model else None
        w = (w_model or {}).get(r["league"], 0.0)
        if pm is not None and w > 0:
            lp = w * np.log(pm) + (1 - w) * np.log(p)
            p = np.exp(lp) / np.exp(lp).sum()
        eligible = r["league"] in ep_leagues
        row = {"match_id": r["match_id"], "date": r["date"], "league": r["league"], "home": r["home"], "away": r["away"], "priced": True}
        best = None
        zs = {"home": curve.zscore((opp_rate or {}).get(r["away"])), "away": curve.zscore((opp_rate or {}).get(r["home"])), "draw": 0.0}
        for j, s in enumerate(SELS):
            ep = float(curve(p[j], zs[s])) if eligible and s != "draw" else 0.0
            row[f"z_{s}"] = round(zs[s], 3)
            p_paid = min(p[j] + ep, 1.0)
            o365 = r.get(f"B365_{s}")
            row[f"fair_{s}"] = round(1.0 / p[j], 2)
            row[f"p_{s}"] = round(float(p[j]), 4)
            row[f"ep_{s}"] = round(ep, 4)
            row[f"min_b365_{s}"] = float(np.ceil(100 * (1 + margin) / p_paid) / 100)
            row[f"min_other_{s}"] = float(np.ceil(100 * (1 + margin) / p[j]) / 100)
            row[f"b365_{s}"] = float(o365) if o365 is not None and np.isfinite(o365) else None
            # other bookmakers in the file already paying the required price (no early payout assumed for them)
            offers = []
            for b, name in BOOKS.items():
                if b in ("B365", "Max"):
                    continue
                o = r.get(f"{b}_{s}")
                if o is not None and np.isfinite(o) and o * p[j] - 1 >= margin:
                    offers.append(f"{name} {o:.2f}")
            mx = r.get(f"Max_{s}")
            row[f"offers_{s}"] = ", ".join(offers)
            row[f"max_{s}"] = float(mx) if mx is not None and np.isfinite(mx) else None
            if o365 is None or not np.isfinite(o365) or o365 <= 1.0:
                continue
            ev = p_paid * o365 - 1.0
            if ev >= margin and ev > 0 and (max_odds is None or o365 <= max_odds) and (best is None or ev > best["ev"]):
                k = kelly_fraction(p_paid, o365)
                stake = min(k * kelly_mult, max_stake_frac) * bankroll
                stake = float(np.floor(stake / round_to) * round_to) if round_to > 0 else stake
                if stake > 0:
                    best = {c: np.nan for c in LEDGER_COLUMNS}
                    best.update({"run_id": run_id, "placed_at": placed_at, "match_id": r["match_id"], "date": pd.Timestamp(r["date"]),
                                 "league": r["league"], "home": r["home"], "away": r["away"], "selection": s, "odds": float(o365),
                                 "p_model": float(pm[j]) if pm is not None else np.nan, "p_market": float(p[j]), "p_pool": float(p[j]),
                                 "ep_bonus": ep, "p_paid": p_paid, "edge": p_paid - float(p[j]), "ev": ev, "kelly_full": k, "stake": stake,
                                 "bankroll_at_placement": bankroll, "settled": False, "ep_unverified": False})
        row["pick"] = best["selection"] if best else ""
        card.append(row)
        if best:
            bets.append(best)
    return pd.DataFrame(bets, columns=LEDGER_COLUMNS), pd.DataFrame(card)


def clv_vs_exchange(ledger: pd.DataFrame, ex: pd.DataFrame, curve: EpCurve | None, ep_leagues, opp_rate: dict | None = None) -> pd.DataFrame:
    """Closing-line value against the exchange close: expected return per euro of the price taken if the close
    is the truth (early payout included for bet365 home/away bets). Returns match_id, selection, closing fair
    price, clv for every ledger row with an exchange close."""
    if len(ledger) == 0 or "fc_home" not in ex.columns:
        return pd.DataFrame(columns=["match_id", "selection", "closing_fair", "clv_ex"])
    m = ledger[["match_id", "selection", "odds", "league", "ep_bonus", "home", "away"]].merge(ex[["match_id", "fc_home", "fc_draw", "fc_away"]], on="match_id", how="inner")
    out = []
    for r in m.to_dict("records"):
        pc = r.get(f"fc_{r['selection']}")
        if pc is None or not np.isfinite(pc):
            continue
        opp = r["away"] if r["selection"] == "home" else r["home"]
        z = curve.zscore((opp_rate or {}).get(opp)) if curve is not None else 0.0
        ep = float(curve(pc, z)) if curve is not None and r["selection"] != "draw" and r["league"] in ep_leagues and float(r.get("ep_bonus") or 0) > 0 else 0.0
        out.append({"match_id": r["match_id"], "selection": r["selection"], "closing_fair": round(1.0 / pc, 3), "clv_ex": (pc + ep) * float(r["odds"]) - 1.0})
    return pd.DataFrame(out, columns=["match_id", "selection", "closing_fair", "clv_ex"])


def verdict(clv: pd.Series, n_min: int = 150) -> dict:
    """Stop-or-continue test on closing-line value vs the exchange close."""
    c = pd.to_numeric(clv, errors="coerce").dropna()
    n = int(len(c))
    if n == 0:
        return {"n": 0, "mean": None, "lower95": None, "share_positive": None, "verdict": "no bets with an exchange close yet"}
    mean, se = float(c.mean()), float(c.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    lo = mean - 1.96 * se if n > 1 else float("nan")
    share = float((c > 0).mean())
    if n < 30:
        v = f"keep testing: {n} of the {n_min} bets needed"
    elif np.isfinite(lo) and lo > 0 and share >= 0.5:
        v = "edge confirmed: closing-line value is positive with 95% confidence"
    elif n >= n_min and mean <= 0:
        v = "stop: no closing-line value after enough bets"
    else:
        v = f"keep testing: {n} of the {n_min} bets needed"
    return {"n": n, "mean": round(mean, 4), "lower95": round(lo, 4) if np.isfinite(lo) else None, "share_positive": round(share, 3), "verdict": v}


# --- slips: core double/treble from the best value legs, and the fun-pot fold -----------------------------------
# Backtest (81 weeks, 2024/25-2026/27, setup "C": quarter Kelly, value >= 2%, odds cap 4.00, plus a EUR 10 six-fold):
# core +12% per euro staked (+-8%), worst drawdown about EUR 1,500 on EUR 10,000; the six-fold of the six shortest
# favourites pays about 4.35, lands about one week in four, and costs about 1 cent per euro on average.

def _leg(r: dict) -> dict:
    return {"match_id": r["match_id"], "league": r["league"], "home": r["home"], "away": r["away"], "date": str(pd.Timestamp(r["date"])),
            "selection": r["selection"], "odds": round(float(r["odds"]), 2), "p": round(float(r["p"]), 4), "ep": round(float(r["ep"]), 4)}


def build_slips(sheet: pd.DataFrame, card: pd.DataFrame, margin: float, bankroll: float, kelly_mult: float, max_odds: float | None,
                fun_pot: float, fun_legs: int, run_id: str, core_multis: bool = True, max_stake_frac: float = 0.05,
                round_to: float = 0.5, fun_max_leg: float = 1.50, multi_max_leg: float | None = None) -> list[dict]:
    """Core double and treble from the sheet's best value legs (one per match, highest value first), Kelly-sized on the
    combined price and combined chance (each leg's exchange chance plus its early-payout chance, matches independent),
    kept only when the combined value clears `margin`. Fun pot: one fold of the `fun_legs` shortest-priced bet365
    favourites (<= fun_max_leg) on the card, staked `fun_pot` regardless of value."""
    slips = []
    if core_multis and len(sheet) >= 2:
        s = sheet[sheet["selection"] != "draw"].sort_values("ev", ascending=False).drop_duplicates("match_id")
        if max_odds:
            s = s[s["odds"] <= max_odds]
        if multi_max_leg:
            s = s[s["odds"] <= multi_max_leg]  # core multiples only from legs at or below this price
        legs_all = [_leg({**r, "p": r["p_market"], "ep": r["ep_bonus"]}) for r in s.head(3).to_dict("records")]
        for n, kind in ((2, "double"), (3, "treble")):
            if len(legs_all) < n:
                continue
            legs = legs_all[:n]
            O = float(np.prod([l["odds"] for l in legs]))
            P = float(np.prod([min(l["p"] + l["ep"], 1.0) for l in legs]))
            ev = P * O - 1.0
            if ev < margin:
                continue
            f = min(kelly_mult * ev / (O - 1.0), max_stake_frac)
            stake = float(np.floor(f * bankroll / round_to) * round_to)
            if stake > 0:
                slips.append({"kind": kind, "pot": "core", "legs": legs, "odds": round(O, 4), "p_paid": round(P, 6), "ev": round(ev, 6), "stake": stake})
    if fun_pot and fun_pot > 0 and len(card):
        cands = []
        for r in card.to_dict("records"):
            if not r.get("priced"):
                continue
            best = None
            for s in ("home", "away"):
                o = r.get(f"b365_{s}")
                if o and o <= fun_max_leg and (best is None or o < best[1]):
                    best = (s, o)
            if best:
                cands.append(_leg({**r, "selection": best[0], "odds": best[1], "p": r[f"p_{best[0]}"], "ep": r[f"ep_{best[0]}"]}))
        cands.sort(key=lambda l: l["odds"])
        if len(cands) >= fun_legs:
            legs = cands[:fun_legs]
            O = float(np.prod([l["odds"] for l in legs]))
            P = float(np.prod([min(l["p"] + l["ep"], 1.0) for l in legs]))
            slips.append({"kind": f"{fun_legs}-fold", "pot": "fun", "legs": legs, "odds": round(O, 4), "p_paid": round(P, 6), "ev": round(P * O - 1, 6), "stake": float(fun_pot)})
    for sl in slips:
        sl["run_id"] = run_id
        sl["status"] = "open"
        sl["first_kickoff"] = min(l["date"] for l in sl["legs"])
    return slips


def settle_slip(slip: dict, results: dict, ep_leagues, lead: int = 2) -> dict | None:
    """`results`: match_id -> (home_goals, away_goals, home_max_lead, away_max_lead). A leg is paid when its side won, or,
    for a home/away leg in an early-payout competition, when it led by `lead` (bet365's offer applies to each leg of a
    multiple). Returns the settled fields, or None while any leg is unplayed or a needed goal timeline is missing."""
    paid = []
    for l in slip["legs"]:
        res = results.get(l["match_id"])
        if res is None or any(v is None for v in res[:2]):
            return None
        hg, ag, hml, aml = res
        won = {"home": hg > ag, "away": ag > hg, "draw": hg == ag}[l["selection"]]
        if not won and l["selection"] != "draw" and (l.get("league_code") or l["league"]) in ep_leagues:
            ml = hml if l["selection"] == "home" else aml
            own = hg if l["selection"] == "home" else ag
            if ml is None or (isinstance(ml, float) and np.isnan(ml)):
                if own >= lead:
                    return None  # could have led by two: wait for the goal timeline
            else:
                won = ml >= lead
        paid.append({**l, "score": f"{hg}–{ag}", "paid": bool(won)})
    allw = all(x["paid"] for x in paid)
    return {"legs": paid, "status": "won" if allw else "lost", "pnl": round(slip["stake"] * (slip["odds"] - 1.0), 2) if allw else -float(slip["stake"])}
