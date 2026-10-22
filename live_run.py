"""Weekly run on the Drive folder files: sheet, settle, dashboard datasets.

    python live_run.py --data <folder with E0_/SP1_/I1_ season files + fixtures.csv> --out <folder> \
        [--ledger <existing ledger.csv>] [--bankroll 1000] [--mode sheet|settle|both]

Writes to --out:
  sheet_<date>.csv, ledger.csv, ratings_<date>.csv, run_<date>.log           (for the Drive folder)
  dash/status.csv, ratings.csv, season.csv, sheet.csv, ledger.csv, market_test.csv  (for the dashboard)
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
for cand in (os.environ.get("FOOTQUANT_PKG"), str(Path(__file__).resolve().parent / "footquant_model"), str(Path(__file__).resolve().parent), "/home/claude/footquant_model"):
    if cand and Path(cand).exists():
        sys.path.insert(0, cand)
        break
from footquant import BacktestConfig, ModelConfig, StakingConfig, ValueConfig  # noqa: E402
from footquant.backtest import walk_forward  # noqa: E402
from footquant.backtest.metrics import outcome_index, rps, log_loss  # noqa: E402
from footquant.data.ingest import DIV_TO_LEAGUE, load_football_data_csv, load_football_data_fixtures  # noqa: E402
from footquant.market.blend import fit_pool_weight  # noqa: E402
from footquant.market.devig import devig  # noqa: E402
from footquant.market.odds_feed import snapshots_from_frame  # noqa: E402
from footquant.models.dixon_coles import DixonColesModel  # noqa: E402
from footquant.paper_trading import Ledger, make_bet_sheet, pooled_fixture  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uefa_data import COMP, OTHER_LEAGUE, load_espn, load_odds_api, team_leagues  # noqa: E402

SHORT = {"ENG-Premier League": "Premier League", "ESP-La Liga": "La Liga", "ITA-Serie A": "Serie A", "GER-Bundesliga": "Bundesliga", "FRA-Ligue 1": "Ligue 1",
         "UEFA-Champions League": "Champions League", "UEFA-Europa League": "Europa League", "UEFA-Conference League": "Conference League", OTHER_LEAGUE: "Europe (other clubs)"}
UEFA_LEAGUES = list(COMP.values())
UEFA_W = 0.01  # weight on the model for the European competitions (no walk-forward test yet; the domestic leagues came out at 0.00-0.01)
HIST_ROWS = [
    {"league": "Premier League", "season": "2017-18", "n_test": 380, "w_model": 0.000, "rps_model": 0.1892, "rps_market": 0.1862, "rps_pool": 0.1862, "source": "Understat xG + football-data mirror"},
    {"league": "Premier League", "season": "2018-19", "n_test": 380, "w_model": 0.000, "rps_model": 0.1842, "rps_market": 0.1855, "rps_pool": 0.1855, "source": "Understat xG + football-data mirror"},
    {"league": "Premier League", "season": "2019-20", "n_test": 380, "w_model": 0.069, "rps_model": 0.1978, "rps_market": 0.1990, "rps_pool": 0.1987, "source": "Understat xG + football-data mirror"},
    {"league": "Premier League", "season": "2020-21 (to Jan)", "n_test": 184, "w_model": 0.222, "rps_model": 0.2127, "rps_market": 0.2115, "rps_pool": 0.2111, "source": "Understat xG + football-data mirror"},
]


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")


def load_history(data: Path):
    frames, odds = [], []
    for f in sorted(glob.glob(str(data / "*_????.csv"))):
        m = re.match(r"([A-Z]+\d)_(\d\d)(\d\d)\.csv$", Path(f).name)
        if not m or m.group(1) not in DIV_TO_LEAGUE:
            continue
        league = DIV_TO_LEAGUE[m.group(1)]
        season = f"20{m.group(2)}-{m.group(3)}"
        mm, oo = load_football_data_csv(f, league, season)
        frames.append(mm)
        odds.append(oo)
    return pd.concat(frames, ignore_index=True), pd.concat(odds, ignore_index=True)


def ledger_from_docs(folder: Path) -> pd.DataFrame:
    """Rebuild the ledger from the Desk's bet documents: open/won/lost bets as recorded, plus watch-list picks
    Miguel marked as placed (at his price and stake)."""
    import json
    from footquant.paper_trading import LEDGER_COLUMNS
    rows = []
    for f in sorted(folder.rglob("*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "match_id" not in d or "selection" not in d:
            continue
        st = d.get("status")
        if st == "watch":
            if not d.get("placed") or not d.get("placed_stake"):
                continue
            odds, stake, run_id = float(d["placed_odds"]), float(d["placed_stake"]), f"placed-{str(d.get('placed_at', ''))[:10]}"
            p_paid = d.get("p_paid"); ev = (float(p_paid) * odds - 1.0) if p_paid else np.nan
        elif st in ("open", "won", "lost"):
            odds, stake, run_id = float(d["odds"]), float(d["stake"]), d.get("run_id")
            ev = d.get("ev")
        else:
            continue
        rows.append({
            "run_id": run_id, "placed_at": d.get("recorded_at") or d.get("placed_at"), "match_id": d["match_id"], "date": pd.Timestamp(d["kickoff"]),
            "league": d.get("league_code") or next((k for k, v in SHORT.items() if v == d.get("league")), d.get("league")),
            "home": d["home"], "away": d["away"], "selection": d["selection"], "odds": odds,
            "p_model": d.get("p_model"), "p_market": d.get("p_market"), "p_pool": d.get("p_pool"), "ep_bonus": d.get("ep_bonus"),
            "p_paid": d.get("p_paid"), "edge": d.get("edge"), "ev": ev, "kelly_full": d.get("kelly_full"), "stake": stake,
            "bankroll_at_placement": d.get("bankroll_at_placement"), "settled": st in ("won", "lost"),
            "won": 1.0 if st == "won" else 0.0 if st == "lost" else np.nan, "pnl": d.get("pnl") if st in ("won", "lost") else np.nan,
            "ep_unverified": bool(d.get("ep_unverified")) or d.get("early_payout") == "unknown",
            "closing_odds": d.get("closing_odds"), "clv": d.get("clv"),
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def bet_on(r):
    return r["home"] if r["selection"] == "home" else r["away"] if r["selection"] == "away" else "Draw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=None, help="existing ledger.csv")
    ap.add_argument("--ledger-docs", default=None, help="folder of bet documents exported from the Desk database (JSON files); builds the ledger from them")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--mode", choices=["sheet", "settle", "both"], default="both")
    ap.add_argument("--now", default=None, help="override 'now' (ISO) for the live/dry-run decision")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--uefa-days", type=int, default=7, help="price European fixtures this many days ahead")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    (out / "dash").mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now()
    today = now.strftime("%Y-%m-%d")
    fh = open(out / f"run_{today}.log", "w")

    hist, odds = load_history(data)
    leagues = sorted(hist["league"].unique())
    log(f"run {now:%Y-%m-%d %H:%M} | {len(hist)} matches in {len(leagues)} leagues: " + ", ".join(f"{SHORT.get(l, l)} {int((hist.league == l).sum())}" for l in leagues), fh)
    log(f"xG available for {int(hist.home_xg.notna().sum())} matches (current-season files only)", fh)

    # --- European competitions: results/fixtures from the ESPN extract, bet365 prices from The Odds API files ---
    uefa_path = data / "uefa_espn.csv"
    if uefa_path.exists():
        u_hist, u_odds, u_fx = load_espn(uefa_path)
        log(f"UEFA: {len(u_hist)} results ({u_hist.date.min():%d %b %Y} to {u_hist.date.max():%d %b %Y}), {len(u_fx)} fixtures to {u_fx.date.max():%d %b}: "
            + ", ".join(f"{SHORT[l]} {int((u_hist.league == l).sum())}" for l in UEFA_LEAGUES), fh)
        b365, api_log = load_odds_api(data, u_fx)
        for line in api_log:
            log("  odds api: " + line, fh)
        if len(b365) == 0:
            log("  no bet365 prices for Europe yet (odds_*.json missing or empty) -> DraftKings prices from ESPN stand in; the desk shows the minimum bet365 price to take", fh)
    else:
        u_hist = pd.DataFrame(columns=list(hist.columns) + ["neutral", "home_max_lead", "away_max_lead"])
        u_odds, u_fx, b365 = odds.iloc[0:0].copy(), pd.DataFrame(columns=["match_id", "date", "league", "season", "home", "away", "neutral"]), odds.iloc[0:0].copy()
        log("UEFA: no uefa_espn.csv in the data folder -> European competitions skipped", fh)
    all_hist = pd.concat([hist, u_hist], ignore_index=True)
    all_hist["neutral"] = all_hist["neutral"].fillna(False).astype(bool) if "neutral" in all_hist.columns else False
    all_odds = pd.concat([odds, u_odds, b365], ignore_index=True)

    mcfg = ModelConfig(xg_weight=0.6, xi=0.003)
    vcfg = ValueConfig(edge_margin=args.margin, markets=frozenset({"1X2"}))
    scfg = StakingConfig()

    # --- pool weight and market test per league ------------------------------------------------
    weights, mt_rows, ratings_rows = {}, [], []
    for lg in leagues:
        part = hist[hist.league == lg]
        start = part.sort_values("date")["date"].iloc[min(380, len(part) - 1)]
        preds = walk_forward(part, mcfg, BacktestConfig(refit_every_days=7, min_train_matches=300), start_date=start)
        o = odds[(odds.market == "1X2") & (~odds.is_closing)].pivot_table(index="match_id", columns="selection", values="odds")
        preds = preds[preds.match_id.isin(o.index)]
        o = o.loc[preds.match_id]
        pk = np.array([devig([h, d, a], "shin") for h, d, a in zip(o.home, o.draw, o.away)])
        pm = preds[["p_home", "p_draw", "p_away"]].values
        y = outcome_index(preds.home_goals.values, preds.away_goals.values)
        w, info = fit_pool_weight(pm, pk, y)
        weights[lg] = w
        mt_rows.append({"league": SHORT.get(lg, lg), "season": f"{start:%b %Y} to {part.date.max():%b %Y} (your files)", "n_test": len(preds), "w_model": round(w, 3),
                        "rps_model": rps(pm, y), "rps_market": rps(pk, y), "rps_pool": rps(np.exp(w * np.log(pm) + (1 - w) * np.log(pk)) / np.exp(w * np.log(pm) + (1 - w) * np.log(pk)).sum(axis=1, keepdims=True), y),
                        "source": "football-data files in the Drive folder"})
        log(f"{SHORT.get(lg, lg)}: weight on model {w:.3f} over {len(preds)} matches (log loss model {info['log_loss_model']:.4f}, bet365 {info['log_loss_market']:.4f})", fh)
        m = DixonColesModel(mcfg)
        p = m.fit(part)
        cur = part[part.season == part.season.max()]
        played = pd.concat([cur.home, cur.away]).value_counts()
        for t, a, d in zip(p.teams, p.attack, p.defence):
            ratings_rows.append({"league": SHORT.get(lg, lg), "team": t, "attack_goals_x": round(float(np.exp(a)), 2), "defence_goals_x": round(float(np.exp(d)), 2), "matches_2627": int(played.get(t, 0))})
        log(f"  fit as of {p.fitted_at.date()}: home advantage x{np.exp(p.home_adv):.2f}, rho {p.rho:+.3f}", fh)
    # joint cross-league fit (domestic files + European matches): one strength offset per league, shared by the UEFA sheet
    joint, league_gaps = None, {}
    if len(u_hist):
        tl = team_leagues(hist, u_hist)
        joint = DixonColesModel(mcfg)
        jp = joint.fit(all_hist, team_league=tl)
        league_gaps = {r.league: float(r.goal_ratio_vs_mean) for r in jp.league_table().itertuples()}
        log("joint fit: " + ", ".join(f"{SHORT.get(l, l)} x{g:.2f}" for l, g in sorted(league_gaps.items(), key=lambda kv: -kv[1])) + f"; home advantage x{np.exp(jp.home_adv):.2f}", fh)
        cur_u = u_hist[u_hist.season == u_hist.season.max()]
        played_u = pd.concat([cur_u.home, cur_u.away]).value_counts()
        for t, a, d in zip(jp.teams, jp.attack, jp.defence):
            if tl.get(t) == OTHER_LEAGUE:
                ratings_rows.append({"league": SHORT[OTHER_LEAGUE], "team": t, "attack_goals_x": round(float(np.exp(a)), 2), "defence_goals_x": round(float(np.exp(d)), 2), "matches_2627": int(played_u.get(t, 0))})
    ratings = pd.DataFrame(ratings_rows)
    ratings.to_csv(out / "dash" / "ratings.csv", index=False)
    ratings.to_csv(out / f"ratings_{today}.csv", index=False)
    pd.DataFrame(HIST_ROWS + mt_rows).round(4).to_csv(out / "dash" / "market_test.csv", index=False)

    # --- fixtures: live sheet if the file holds future matches in our leagues, else dry run -------
    fx, fx_odds = load_football_data_fixtures(data / "fixtures.csv", hist.season.max())
    fx = fx[fx.league.isin(leagues)]
    live = fx[fx.date > now]
    sheet_frames = []
    if args.mode in ("sheet", "both"):
        if len(live):
            snaps = snapshots_from_frame(fx_odds[fx_odds.match_id.isin(live.match_id)])
            run_id = f"sheet-{today}"
            frames = []
            for lg in leagues:
                lf = live[live.league == lg]
                if len(lf) == 0:
                    continue
                frames.append(make_bet_sheet(hist, lf, [s for s in snaps if s.match_id in set(lf.match_id)], weights[lg], args.bankroll, mcfg, vcfg, scfg, run_id=run_id))
            sheet = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            sheet_run = f"live sheet of {today}: {len(live)} fixtures priced"
            log(f"LIVE: {len(live)} upcoming fixtures in our leagues -> {len(sheet)} bets", fh)
        else:
            frames = []
            for lg in leagues:
                part = hist[hist.league == lg]
                cut = part.date.max().normalize() - pd.Timedelta(days=6)
                last = part[part.date >= cut]
                snaps = snapshots_from_frame(odds[odds.match_id.isin(last.match_id) & (~odds.is_closing)])
                frames.append(make_bet_sheet(part[part.date < cut], last[["match_id", "date", "league", "home", "away"]], snaps, weights[lg], args.bankroll, mcfg, vcfg, scfg, as_of=cut, run_id=f"dryrun-{cut:%Y-%m-%d}"))
            sheet = pd.concat(frames, ignore_index=True)
            sheet_run = "dry run on the last completed round (no upcoming fixtures in the fixtures file yet)"
            log(f"no upcoming fixtures for our leagues in fixtures.csv (it covers {fx.date.min():%d %b}-{fx.date.max():%d %b} / {len(fx)} rows) -> dry run on the last round: {len(sheet)} bets", fh)
        # European fixtures in the window, priced with bet365 when the odds files are there, else DraftKings (ESPN)
        price_source = {}
        watch_rows = []
        u_live = u_fx[(u_fx.date > now) & (u_fx.date <= now + pd.Timedelta(days=args.uefa_days))] if len(u_fx) else u_fx
        if len(u_live) and joint is not None:
            cand = pd.concat([b365, u_odds[(~u_odds.is_closing) & u_odds.match_id.isin(u_live.match_id)]], ignore_index=True)
            keep = []
            for mid, part in cand.groupby("match_id"):
                books = set(part.bookmaker)
                use = "bet365" if "bet365" in books else sorted(books)[0]
                price_source[mid] = use
                keep.append(part[part.bookmaker == use])
            snaps = snapshots_from_frame(pd.concat(keep, ignore_index=True), bookmaker=None) if keep else []
            u_sheet = make_bet_sheet(all_hist, u_live, snaps, UEFA_W, args.bankroll, mcfg, vcfg, scfg, run_id=f"uefa-{today}",
                                     models={lg: joint for lg in UEFA_LEAGUES})
            sheet_pick = {r["match_id"]: (r["selection"], float(r["stake"]), float(r["ev"])) for r in u_sheet.to_dict("records")}
            # only bets priced by bet365 go into the ledger; stand-in prices make a watch list instead
            u_sheet = u_sheet[u_sheet.match_id.map(lambda m: price_source.get(m) == "bet365")]
            sheet = pd.concat([sheet, u_sheet], ignore_index=True) if len(sheet) else u_sheet
            # watch list: for every priced European fixture, the selection closest to value and the bet365 price that makes it one
            by_match = {sn.match_id: sn for sn in snaps if sn.market == "1X2"}
            seed = 7000
            for rec in u_live.to_dict("records"):
                sn = by_match.get(rec["match_id"])
                if sn is None:
                    continue
                f = pooled_fixture(joint, sn, rec["home"], rec["away"], UEFA_W, max_goals=mcfg.max_goals, n_sims=vcfg.n_sims, seed=seed,
                                   neutral=bool(rec.get("neutral", False)), value_cfg=vcfg)
                seed += 1
                if price_source.get(rec["match_id"]) == "bet365":
                    continue  # priced properly: it is either on the sheet or not worth a bet
                best = None
                pick = sheet_pick.get(rec["match_id"])
                for j, sel in enumerate(("home", "draw", "away")):
                    bonus = float(f["bonus"][j]) * vcfg.early_payout_scale
                    p_paid = min(float(f["p_pool"][j]) + bonus, 1.0)
                    min_odds = float(np.ceil(100.0 / round(p_paid, 4)) / 100.0)
                    cur = float(sn.prices[sel])
                    cand_row = {"match_id": rec["match_id"], "date": rec["date"], "league": rec["league"], "home": rec["home"], "away": rec["away"],
                                "selection": sel, "p_paid": p_paid, "p_market": float(f["p_market"][j]), "ep_bonus": bonus, "min_odds": min_odds,
                                "odds": cur, "source": price_source.get(rec["match_id"], sn.bookmaker), "ratio": cur / min_odds,
                                "stake": pick[1] if pick and pick[0] == sel else 0.0, "ev": pick[2] if pick and pick[0] == sel else None,
                                "edge": p_paid - float(f["p_market"][j])}
                    if pick:
                        if pick[0] == sel:
                            best = cand_row
                    elif best is None or cand_row["ratio"] > best["ratio"]:
                        best = cand_row
                watch_rows.append(best)
            n_b365 = sum(1 for m in u_live.match_id if price_source.get(m) == "bet365")
            log(f"UEFA: {len(u_live)} fixtures in the next {args.uefa_days} days ({n_b365} with bet365 prices, {len(price_source) - n_b365} with stand-in prices) -> {len(u_sheet)} bets on the sheet, {len(watch_rows)} on the watch list ({sum(1 for w in watch_rows if w['stake'] > 0)} showing value at the stand-in price)", fh)
            sheet_run += f"; Europe: {len(u_live)} fixtures priced" + ("" if n_b365 == len(price_source) else " on stand-in prices")
        elif len(u_fx):
            log(f"UEFA: no European fixtures in the next {args.uefa_days} days (next on {u_fx.date.min():%d %b})", fh)
        sheet.to_csv(out / f"sheet_{today}.csv", index=False)
    else:
        sheet = pd.DataFrame(columns=["match_id"])
        sheet_run = "no sheet produced in this run"
        price_source = {}
        watch_rows = []

    # --- ledger: existing + new sheet, settle what has results ---------------------------------
    ledger_path = out / "ledger.csv"
    if args.ledger_docs and Path(args.ledger_docs).exists():
        ledger_from_docs(Path(args.ledger_docs)).to_csv(ledger_path, index=False)
        log(f"ledger rebuilt from {len(list(Path(args.ledger_docs).rglob('*.json')))} Desk documents", fh)
    elif args.ledger and Path(args.ledger).exists():
        import shutil
        shutil.copy(args.ledger, ledger_path)
    led = Ledger(ledger_path)
    if len(sheet):
        n_new = led.append(sheet)
        log(f"ledger: {n_new} new bets recorded", fh)
    if args.mode in ("settle", "both"):
        n = led.settle(all_hist, all_odds)
        log(f"ledger: {n} bets settled", fh)
    rep = led.report()
    log("ledger report: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()), fh)

    # --- dashboard datasets --------------------------------------------------------------------
    def plain(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["league"] = df["league"].map(lambda l: SHORT.get(l, l))
        df["sort"] = df.date.dt.strftime("%Y-%m-%d %H:%M")
        df["day"] = df.date.dt.strftime("%a %d %b")
        df["kickoff"] = df.date.dt.strftime("%H:%M")
        df["match"] = df.home + " – " + df.away
        df["bet_on"] = df.apply(bet_on, axis=1)
        return df

    if len(sheet):
        s = plain(sheet)
        s["note"] = np.where(s.ep_bonus > 0, "bet365 pays early if they lead by 2", "")
        s[["league", "match_id", "run_id", "sort", "day", "kickoff", "match", "bet_on", "odds", "stake", "note", "edge", "ev"]].round(4).to_csv(out / "dash" / "sheet.csv", index=False)
    else:
        pd.DataFrame(columns=["league", "match_id", "run_id", "sort", "day", "kickoff", "match", "bet_on", "odds", "stake", "note", "edge", "ev"]).to_csv(out / "dash" / "sheet.csv", index=False)

    l = plain(led.df) if len(led.df) else pd.DataFrame()
    if len(l):
        res = all_hist.set_index("match_id")
        def score(r):
            if r.match_id in res.index:
                x = res.loc[r.match_id]
                return f"{int(x.home_goals)}–{int(x.away_goals)}"
            return ""
        l["score"] = l.apply(score, axis=1)
        def ep(r):
            if r.match_id not in res.index or not r.settled:
                return ""
            x = res.loc[r.match_id]
            hg, ag = int(x.home_goals), int(x.away_goals)
            nat = (hg > ag) if r.selection == "home" else (ag > hg) if r.selection == "away" else (hg == ag)
            if r.won == 1.0 and nat:
                return "not needed"
            if r.won == 1.0:
                return "yes"
            return "unknown (needs goal times)" if r.ep_unverified else "no"
        l["early_payout"] = l.apply(ep, axis=1)
        l["result"] = np.where(l.won == 1.0, "Won", np.where(l.won == 0.0, "Lost", "Open"))
        l[["league", "match_id", "run_id", "sort", "day", "match", "bet_on", "odds", "stake", "score", "result", "pnl", "early_payout", "closing_odds", "clv"]].round(4).to_csv(out / "dash" / "ledger.csv", index=False)

    # --- app documents (Footquant app db): one document per bet + a status document --------------
    import hashlib, json
    def _f(v, nd=4):
        return round(float(v), nd) if v is not None and pd.notna(v) else None
    docs = []
    src_path = out / "price_source.json"
    sources = json.load(open(src_path)) if src_path.exists() else {}
    sources.update(price_source)
    json.dump(sources, open(src_path, "w"), indent=0)
    res_all = all_hist.set_index("match_id")
    if len(led.df):
        for r in led.df.to_dict("records"):
            doc_id = hashlib.sha1(f"{r['match_id']}|{r['selection']}".encode()).hexdigest()[:16]
            kick = pd.Timestamp(r["date"])
            hg = ag = None
            if r["match_id"] in res_all.index:
                x = res_all.loc[r["match_id"]]
                hg, ag = int(x.home_goals), int(x.away_goals)
            src = sources.get(r["match_id"], "bet365")
            min_odds = round(float(np.ceil(100.0 / round(float(r["p_paid"]), 4)) / 100.0), 2) if float(r.get("p_paid") or 0) > 0 else None
            status = "won" if r.get("won") == 1.0 else "lost" if r.get("won") == 0.0 else "open"
            nat = None
            if hg is not None:
                nat = (hg > ag) if r["selection"] == "home" else (ag > hg) if r["selection"] == "away" else (hg == ag)
            ep = ""
            if status == "won":
                ep = "not needed" if nat else "yes"
            elif status == "lost":
                ep = "unknown" if r.get("ep_unverified") else "no"
            docs.append({
                "id": doc_id, "match_id": r["match_id"], "run_id": r["run_id"], "league": SHORT.get(r["league"], r["league"]),
                "kickoff": kick.strftime("%Y-%m-%dT%H:%M:00"), "day": kick.strftime("%Y-%m-%d"),
                "home": r["home"], "away": r["away"], "match": f"{r['home']} – {r['away']}", "bet_on": bet_on(r), "selection": r["selection"],
                "odds": round(float(r["odds"]), 2), "stake": round(float(r["stake"]), 2), "edge": round(float(r["edge"]), 4), "ev": round(float(r["ev"]), 4),
                "ep_bonus": round(float(r["ep_bonus"]), 4), "note": "bet365 pays early if they lead by 2" if float(r["ep_bonus"]) > 0 else "",
                "status": status, "score": f"{hg}–{ag}" if hg is not None and status != "open" else "",
                "pnl": round(float(r["pnl"]), 2) if status != "open" else None, "early_payout": ep,
                "closing_odds": round(float(r["closing_odds"]), 2) if pd.notna(r.get("closing_odds")) else None,
                "clv": round(float(r["clv"]), 4) if pd.notna(r.get("clv")) else None,
                "price_source": src, "min_odds": min_odds,
                # the rest makes the document a complete ledger row (the Desk database is the record of bets)
                "league_code": r["league"], "recorded_at": r.get("placed_at") if isinstance(r.get("placed_at"), str) else None, "p_model": _f(r.get("p_model")), "p_market": _f(r.get("p_market")),
                "p_pool": _f(r.get("p_pool")), "p_paid": _f(r.get("p_paid")), "kelly_full": _f(r.get("kelly_full")),
                "bankroll_at_placement": _f(r.get("bankroll_at_placement")), "ep_unverified": bool(r.get("ep_unverified")),
            })
    staked_ids = set(led.df["match_id"]) if len(led.df) else set()
    for r in watch_rows:
        if r["match_id"] in staked_ids:
            continue  # it made the sheet proper
        kick = pd.Timestamp(r["date"])
        docs.append({
            "id": hashlib.sha1(f"{r['match_id']}|{r['selection']}|watch".encode()).hexdigest()[:16], "match_id": r["match_id"], "run_id": f"watch-{today}",
            "league": SHORT.get(r["league"], r["league"]), "league_code": r["league"], "kickoff": kick.strftime("%Y-%m-%dT%H:%M:00"), "day": kick.strftime("%Y-%m-%d"),
            "home": r["home"], "away": r["away"], "match": f"{r['home']} – {r['away']}", "bet_on": bet_on(r), "selection": r["selection"],
            "p_market": round(r["p_market"], 4),
            "odds": round(r["odds"], 2), "stake": round(r["stake"], 2), "edge": round(r["edge"], 4), "ev": (round(r["ev"], 4) if r["ev"] is not None else None),
            "ep_bonus": round(r["ep_bonus"], 4), "p_paid": round(r["p_paid"], 4),
            "min_odds": r["min_odds"], "price_source": r["source"], "status": "watch",
            "note": ("model and market agree: nothing to take at any price" if r["edge"] < args.margin else (f"value at {r['source']}'s {r['odds']:.2f}" if r["stake"] > 0 else f"{r['source']} shows {r['odds']:.2f}") + f"; take it only if bet365 pays at least {r['min_odds']:.2f}") + (" · bet365 pays early if they lead by 2" if r["ep_bonus"] > 0 else ""),
            "score": "", "pnl": None, "early_payout": "", "closing_odds": None, "clv": None,
        })
    json.dump(docs, open(out / "dash" / "bets.json", "w"), indent=1)
    uefa_loaded = ", ".join(SHORT[l] for l in UEFA_LEAGUES) if len(u_hist) else ""
    status_doc = {"as_of": now.strftime("%Y-%m-%dT%H:%M:00"), "history_matches": int(len(all_hist)), "history_to": str(all_hist.date.max().date()),
                  "leagues_loaded": ", ".join(SHORT.get(l, l) for l in leagues) + (", " + uefa_loaded if uefa_loaded else ""),
                  "pool_weights": "; ".join(f"{SHORT.get(l, l)} {weights[l]:.3f}" for l in leagues) + (f"; Europe {UEFA_W:.3f} (set)" if len(u_hist) else ""),
                  "uefa_prices": ("bet365" if len(b365) else "stand-in (DraftKings via ESPN) until the Odds API key is in the download script") if len(u_hist) else "not loaded",
                  "fixtures_priced": int(len(live)), "fixtures_file_covers": f"{fx.date.min():%d %b} to {fx.date.max():%d %b}" if len(fx) else "no matches in our leagues",
                  "sheet_run": sheet_run, "bankroll": args.bankroll, "kelly": scfg.kelly_mode, "n_settled": int(rep.get("n_settled", 0))}
    json.dump(status_doc, open(out / "dash" / "status.json", "w"), indent=1)
    # Desk database write plan: one JSON file per document plus batch files (<= 50 writes each) for the ArtifactData tool.
    # New documents -> set; changed documents -> update (if_version filled from versions.json when given, else null).
    docs_dir = out / "dash" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.ledger_docs and Path(args.ledger_docs).exists():
        for f in Path(args.ledger_docs).rglob("*.json"):
            try:
                d = json.load(open(f, encoding="utf-8"))
                if isinstance(d, dict) and "match_id" in d:
                    existing[f.stem] = d
            except Exception:
                pass
    versions = {}
    vf = Path(args.ledger_docs) / "versions.json" if args.ledger_docs else None
    if vf and vf.exists():
        versions = json.load(open(vf))
    user_fields = {"placed", "placed_odds", "placed_stake", "placed_at"}
    writes = []
    for d in docs:
        doc = {k: v for k, v in d.items() if k != "id"}
        old_doc = existing.get(d["id"])
        if old_doc is None:
            doc["placed"] = False
            json.dump(doc, open(docs_dir / f"{d['id']}.json", "w"), ensure_ascii=False)
            writes.append({"op": "set", "collection": "bets", "doc_id": d["id"], "file_path": str((docs_dir / f"{d['id']}.json").resolve())})
        else:
            cmp_old = {k: v for k, v in old_doc.items() if k not in user_fields}
            if cmp_old != doc:
                if os.environ.get("FQ_DEBUG"):
                    print("DIFF", d["id"], {k: (cmp_old.get(k), doc.get(k)) for k in set(cmp_old) | set(doc) if cmp_old.get(k) != doc.get(k)})
                json.dump(doc, open(docs_dir / f"{d['id']}.json", "w"), ensure_ascii=False)
                writes.append({"op": "update", "collection": "bets", "doc_id": d["id"], "file_path": str((docs_dir / f"{d['id']}.json").resolve()),
                               "if_version": versions.get(d["id"])})
    json.dump(status_doc, open(docs_dir / "status.json", "w"), ensure_ascii=False)
    writes.append({"op": "set", "collection": "meta", "doc_id": "status", "file_path": str((docs_dir / "status.json").resolve()),
                   "if_version": versions.get("meta/status")})
    for i in range(0, len(writes), 50):
        json.dump(writes[i:i + 50], open(out / "dash" / f"db_batch_{i // 50 + 1}.json", "w"), indent=0)
    n_set = sum(1 for w in writes if w["op"] == "set") - 1
    log(f"desk database: {n_set} new documents, {len(writes) - 1 - n_set} to update, {len(docs) - len(writes) + 1} unchanged -> dash/db_batch_*.json "
        f"({(len(writes) + 49) // 50} batch file(s){'; fill in if_version from the listing' if not versions else ''})", fh)
    log(f"app documents: {len(docs)} bets ({sum(d['status'] == 'open' for d in docs)} open, {sum(d['status'] == 'watch' for d in docs)} to watch) -> dash/bets.json, dash/status.json", fh)

    cur = hist[hist.season == hist.season.max()].copy()
    cur["league"] = cur.league.map(lambda x: SHORT.get(x, x))
    cur["day"] = cur.date.dt.strftime("%Y-%m-%d")
    cur[["league", "match_id", "day", "home", "away", "home_goals", "away_goals", "home_xg", "away_xg"]].round(2).to_csv(out / "dash" / "season.csv", index=False)

    st = {
        "as_of": today, "history_matches": len(hist), "history_from": str(hist.date.min().date()), "history_to": str(hist.date.max().date()),
        "xg_matches": int(hist.home_xg.notna().sum()), "leagues_loaded": ", ".join(SHORT.get(l, l) for l in leagues),
        "pool_weights": "; ".join(f"{SHORT.get(l, l)} {weights[l]:.3f}" for l in leagues),
        "fixtures_priced": int(len(live)), "fixtures_file_covers": f"{fx.date.min():%d %b} to {fx.date.max():%d %b}" if len(fx) else "no matches in our leagues",
        "bankroll": args.bankroll, "kelly": scfg.kelly_mode, "edge_margin": args.margin, "sheet_run": sheet_run,
        "n_settled": rep.get("n_settled", 0), "leagues_waiting": "", "uefa_note": (f"Champions, Europa and Conference League: {len(u_hist)} results in the model; prices " + ("from bet365" if len(b365) else "stand-in until the Odds API key is set")) if len(u_hist) else "Champions, Europa and Conference League: uefa_espn.csv missing",
    }
    pd.DataFrame([st]).to_csv(out / "dash" / "status.csv", index=False)
    log("done", fh)
    fh.close()


if __name__ == "__main__":
    main()
