"""Walk-forward check of the production pricing on the domestic history.

For every match since the first ~380 of each league file: the model's out-of-sample probabilities (same walk-forward as
live_run), the pooled probabilities with the league's fitted weight, the simulated early-payout increments, bet365's
pre-match and closing 1X2 prices, Pinnacle's pre-match and closing prices, the result, the half-time lead and - when
data/domestic_espn.csv is there - the realised maximum lead of each side from the goal timeline.

Writes <out>/underdog_rows.csv, one row per (match, selection). analyse with underdog_report.py.
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
HERE = Path(__file__).resolve().parent
for cand in (os.environ.get("FOOTQUANT_PKG"), str(HERE / "footquant_model"), "/home/claude/footquant_model", str(HERE)):
    if cand and (Path(cand) / "footquant").exists():
        sys.path.insert(0, cand)
        break
sys.path.insert(0, str(HERE))

from footquant.backtest import walk_forward  # noqa: E402
from footquant.backtest.metrics import outcome_index  # noqa: E402
from footquant.config import BacktestConfig, ModelConfig, ValueConfig  # noqa: E402
from footquant.data.ingest import DIV_TO_LEAGUE, read_football_data_csv  # noqa: E402
from footquant.market.blend import fit_pool_weight, pool_with_market  # noqa: E402
from footquant.market.devig import devig  # noqa: E402
from domestic_data import domestic_timelines  # noqa: E402
from live_run import load_history  # noqa: E402

SEL = ("home", "draw", "away")


def raw_prices(data: Path) -> pd.DataFrame:
    """Pinnacle pre-match/closing and bet365 closing prices plus half-time goals, keyed like load_football_data_csv."""
    rows = []
    for f in sorted(glob.glob(str(data / "*_????.csv"))):
        m = re.match(r"([A-Z]+\d)_(\d\d)(\d\d)\.csv$", Path(f).name)
        if not m or m.group(1) not in DIV_TO_LEAGUE:
            continue
        league, season = DIV_TO_LEAGUE[m.group(1)], f"20{m.group(2)}-{m.group(3)}"
        raw = read_football_data_csv(f).dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        date = pd.to_datetime(raw["Date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
        for i, (d, h, a) in enumerate(zip(date, raw["HomeTeam"].astype(str).str.strip(), raw["AwayTeam"].astype(str).str.strip())):
            r = raw.iloc[i]
            rows.append({"match_id": f"{league}|{season}|{d.date()}|{h}|{a}", "div": m.group(1), "date_only": d.date(),
                         "ht_home": r.get("HTHG"), "ht_away": r.get("HTAG"),
                         "ps_h": r.get("PSH"), "ps_d": r.get("PSD"), "ps_a": r.get("PSA"),
                         "psc_h": r.get("PSCH"), "psc_d": r.get("PSCD"), "psc_a": r.get("PSCA"),
                         "b365c_h": r.get("B365CH"), "b365c_d": r.get("B365CD"), "b365c_a": r.get("B365CA")})
    out = pd.DataFrame(rows)
    for c in out.columns:
        if c.startswith(("ps", "b365", "ht")):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.set_index("match_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sims", type=int, default=20000)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist, odds = load_history(data)
    prices = raw_prices(data)
    leads = domestic_timelines(data, hist)
    print(f"{len(hist)} matches, timelines for {len(leads)}", flush=True)
    mcfg = ModelConfig(xg_weight=0.6, xi=0.003)
    vcfg = ValueConfig(edge_margin=0.01, markets=frozenset({"1X2"}))
    pre = odds[(odds.market == "1X2") & (~odds.is_closing)].pivot_table(index="match_id", columns="selection", values="odds")
    rows = []
    seed = 100
    for lg, part in hist.groupby("league"):
        part = part.sort_values("date")
        start = part["date"].iloc[min(380, len(part) - 1)]
        preds = walk_forward(part, mcfg, BacktestConfig(refit_every_days=7, min_train_matches=300), start_date=start)
        preds = preds[preds.match_id.isin(pre.index)]
        o = pre.loc[preds.match_id]
        pk = np.array([devig([h, d, a], "shin") for h, d, a in zip(o.home, o.draw, o.away)])
        pm = preds[["p_home", "p_draw", "p_away"]].values
        y = outcome_index(preds.home_goals.values, preds.away_goals.values)
        w, info = fit_pool_weight(pm, pk, y)
        print(f"{lg}: {len(preds)} matches, w={w:.3f}", flush=True)
        for rec in preds.to_dict("records"):
            mid = rec["match_id"]
            ov = [float(pre.loc[mid, "home"]), float(pre.loc[mid, "draw"]), float(pre.loc[mid, "away"])]
            pooled, ep = pool_with_market(rec, ov, w, None, mcfg.max_goals, args.n_sims, seed, value_cfg=vcfg)
            seed += 1
            pr = prices.loc[mid] if mid in prices.index else None
            res = outcome_index(np.array([rec["home_goals"]]), np.array([rec["away_goals"]]))[0]
            ml = leads.get(mid, (np.nan, np.nan))
            for j, sel in enumerate(SEL):
                bonus = float(ep[f"p_{sel}_bonus"]) if sel != "draw" else 0.0
                ht_lead = np.nan
                if pr is not None and not np.isnan(pr["ht_home"]):
                    ht_lead = (pr["ht_home"] - pr["ht_away"]) if sel == "home" else (pr["ht_away"] - pr["ht_home"]) if sel == "away" else np.nan
                rows.append({
                    "league": lg, "match_id": mid, "date": rec["date"], "home": rec["home"], "away": rec["away"], "selection": sel,
                    "w": w, "p_model": float(rec[f"p_{sel}"]), "p_market": float(pooled[f"m_{sel}"]), "p_pool": float(pooled[f"p_{sel}"]),
                    "bonus": bonus, "odds": ov[j], "won": int(res == j),
                    "max_lead": ml[0] if sel == "home" else ml[1] if sel == "away" else np.nan, "ht_lead": ht_lead,
                    "ps": float(pr[f"ps_{sel[0]}"]) if pr is not None else np.nan,
                    "psc": float(pr[f"psc_{sel[0]}"]) if pr is not None else np.nan,
                    "b365c": float(pr[f"b365c_{sel[0]}"]) if pr is not None else np.nan,
                    "psc_h": float(pr["psc_h"]) if pr is not None else np.nan, "psc_d": float(pr["psc_d"]) if pr is not None else np.nan, "psc_a": float(pr["psc_a"]) if pr is not None else np.nan,
                    "b365c_h": float(pr["b365c_h"]) if pr is not None else np.nan, "b365c_d": float(pr["b365c_d"]) if pr is not None else np.nan, "b365c_a": float(pr["b365c_a"]) if pr is not None else np.nan,
                })
    df = pd.DataFrame(rows)
    # fair closing probabilities (Shin) for CLV
    for src in ("psc", "b365c"):
        fair = np.full(len(df), np.nan)
        ok = df[[f"{src}_h", f"{src}_d", f"{src}_a"]].notna().all(axis=1).values
        for i in np.where(ok)[0]:
            r = df.iloc[i]
            f = devig([r[f"{src}_h"], r[f"{src}_d"], r[f"{src}_a"]], "shin")
            fair[i] = f[SEL.index(r["selection"])]
        df[f"fair_{src}"] = fair
    df.to_csv(out / "underdog_rows.csv", index=False)
    print(f"wrote {len(df)} rows -> {out / 'underdog_rows.csv'}")


if __name__ == "__main__":
    main()
