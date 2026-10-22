"""Read underdog_rows.csv (from underdog_check.py) and answer: does the pricing over-rate underdogs, and what rule fixes it?"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 100.0]
LABELS = ["<1.50", "1.50-2.00", "2.00-2.50", "2.50-3.00", "3.00-4.00", "4.00-6.00", "6.00-10.0", "10.0+"]


def se(x):
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main(path: str, margin: float = 0.01) -> None:
    df = pd.read_csv(path, parse_dates=["date"])
    df["band"] = pd.cut(df["odds"], BANDS, labels=LABELS)
    df["p_paid"] = (df["p_pool"] + df["bonus"]).clip(upper=1.0)
    df["ep_hit"] = ((df["max_lead"] >= 2) & (df["won"] == 0)).astype(float).where(df["max_lead"].notna())
    df["paid_out"] = ((df["won"] == 1) | (df["max_lead"] >= 2)).astype(float).where(df["max_lead"].notna() | (df["won"] == 1))
    n_matches = df["match_id"].nunique()
    print(f"{n_matches} matches, {len(df)} selections, timelines for {int(df['max_lead'].notna().sum() / 2)} matches\n")

    print("1. CALIBRATION by bet365 pre-match price band (all three selections of every match)")
    g = df.groupby("band", observed=True)
    cal = pd.DataFrame({
        "n": g.size(), "actual%": 100 * g["won"].mean(), "1/odds%": 100 * (1 / g["odds"].mean()),
        "market(Shin)%": 100 * g["p_market"].mean(), "model%": 100 * g["p_model"].mean(), "pool%": 100 * g["p_pool"].mean(),
    }).round(1)
    cal["actual-market (pts)"] = (cal["actual%"] - cal["market(Shin)%"]).round(1)
    cal["actual-model (pts)"] = (cal["actual%"] - cal["model%"]).round(1)
    print(cal.to_string(), "\n")

    print("2. EARLY PAYOUT (home/away selections): simulated bonus vs realised 'led by two, did not win'")
    ha = df[df["selection"] != "draw"]
    g = ha.groupby("band", observed=True)
    ep = pd.DataFrame({"n": g.size(), "simulated bonus (pts)": 100 * g["bonus"].mean(), "realised (pts)": 100 * g["ep_hit"].mean(),
                       "realised se": 100 * g["ep_hit"].apply(se), "led by 2 %": 100 * g["max_lead"].apply(lambda x: (x >= 2).mean())}).round(2)
    print(ep.to_string(), "\n")

    def bets(frame, margin=0.01, cap=None, pick="ev", bonus_scale=None, min_ev_no_bonus=None):
        f = frame.copy()
        if bonus_scale is not None:
            f["bonus"] = f["bonus"] * f["band"].map(bonus_scale).astype(float).fillna(1.0)
            f["p_paid"] = (f["p_pool"] + f["bonus"]).clip(upper=1.0)
        f["edge"] = f["p_paid"] - f["p_market"]
        f["ev"] = f["p_paid"] * f["odds"] - 1
        f["ev0"] = f["p_pool"] * f["odds"] - 1
        ok = (f["edge"] >= margin) & (f["ev"] > 0)
        if cap:
            ok &= f["odds"] <= cap
        if min_ev_no_bonus is not None:
            ok &= f["ev0"] >= min_ev_no_bonus
        f = f[ok]
        key = "ev" if pick == "ev" else "edge"
        f = f.sort_values(key, ascending=False).drop_duplicates("match_id")
        f["pnl"] = np.where(f["paid_out"] == 1, f["odds"] - 1, -1.0)
        f["clv_ps"] = f["fair_psc"] * f["odds"] - 1
        f["clv_b365"] = f["fair_b365c"] * f["odds"] - 1
        f["ev_close"] = (f["fair_psc"] + f["bonus"]) * f["odds"] - 1
        return f

    def table(b, by="band"):
        g = b.groupby(by, observed=True)
        t = pd.DataFrame({
            "bets": g.size(), "hit% (FT)": 100 * g["won"].mean(), "paid% (incl EP)": 100 * g["paid_out"].mean(), "expected%": 100 * g["p_paid"].mean(),
            "avg odds": g["odds"].mean(), "yield%": 100 * g["pnl"].mean(), "yield se": 100 * g["pnl"].apply(se),
            "CLV% Pinnacle": 100 * g["clv_ps"].mean(), "CLV se": 100 * g["clv_ps"].apply(se), "EV@close+bonus%": 100 * g["ev_close"].mean(),
        }).round(1)
        return t

    def total(b, name):
        return {"rule": name, "bets": len(b), "per match%": round(100 * len(b) / n_matches, 1), "avg odds": round(b["odds"].mean(), 2),
                "hit% (FT)": round(100 * b["won"].mean(), 1), "paid%": round(100 * b["paid_out"].mean(), 1), "expected%": round(100 * b["p_paid"].mean(), 1),
                "yield%": round(100 * b["pnl"].mean(), 1), "yield se": round(100 * se(b["pnl"]), 1),
                "CLV%": round(100 * b["clv_ps"].mean(), 1), "CLV se": round(100 * se(b["clv_ps"]), 1), "EV@close+bonus%": round(100 * b["ev_close"].mean(), 1)}

    print(f"3. THE PRODUCTION RULE on history (edge >= {margin:.0%} pts, EV > 0, best EV per match, flat stakes, early payout credited from the timeline)")
    prod = bets(df, margin)
    print(table(prod).to_string(), "\n")
    print("   by selection:"); print(table(prod, "selection").to_string(), "\n")

    print("4. ALTERNATIVE RULES (totals)")
    rows = [total(prod, "production: edge>=1pt, best EV"),
            total(bets(df, 0.02), "edge>=2pts"), total(bets(df, 0.03), "edge>=3pts"),
            total(bets(df, margin, cap=3.0), "cap 3.00"), total(bets(df, margin, cap=2.5), "cap 2.50"), total(bets(df, margin, cap=2.0), "cap 2.00"),
            total(bets(df, margin, pick="edge"), "pick by edge (pts), not EV"),
            total(bets(df, margin, min_ev_no_bonus=-0.02), "EV without bonus >= -2%"), total(bets(df, margin, min_ev_no_bonus=0.0), "EV without bonus >= 0"),
            ]
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    print("5. WHERE DOES THE EDGE COME FROM in the production picks")
    prod["vig_share"] = 1 - prod["p_market"] * prod["odds"]
    prod["bonus_x_odds"] = prod["bonus"] * prod["odds"]
    prod["model_x_odds"] = (prod["p_pool"] - prod["p_market"]) * prod["odds"]
    print(prod.groupby("band", observed=True)[["vig_share", "bonus_x_odds", "model_x_odds", "ev"]].mean().mul(100).round(1).rename(columns={
        "vig_share": "price below fair% (Shin)", "bonus_x_odds": "early payout x odds%", "model_x_odds": "model view x odds%", "ev": "EV%"}).to_string(), "\n")
    prod.to_csv(path.replace("rows", "prod_bets"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.01)
