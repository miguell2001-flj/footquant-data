"""Goal timelines for the five domestic leagues (data/domestic_espn.csv, written by fetch.py from the ESPN scoreboard):
who led by two goals at some point. Joined to the football-data.co.uk matches by league, date (+-1 day) and team name."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footquant.data.ingest import DIV_TO_LEAGUE
from uefa_data import _norm, match_name

# football-data.co.uk spellings that the generic matcher gets wrong or misses -> ESPN display names
FD_TO_ESPN = {
    "Man City": "Manchester City", "Man United": "Manchester United", "Wolves": "Wolverhampton Wanderers", "Nott'm Forest": "Nottingham Forest",
    "Paris SG": "Paris Saint-Germain", "St Etienne": "Saint-Étienne", "Le Havre": "Le Havre AC", "Reims": "Stade de Reims", "Rennes": "Stade Rennais",
    "Inter": "Internazionale", "Milan": "AC Milan", "Roma": "AS Roma", "Verona": "Hellas Verona",
    "Celta": "Celta Vigo", "Espanol": "Espanyol", "Ath Madrid": "Atlético Madrid", "Ath Bilbao": "Athletic Club", "Sociedad": "Real Sociedad",
    "Betis": "Real Betis", "Alaves": "Alavés", "Leganes": "Leganés", "Vallecano": "Rayo Vallecano", "Valladolid": "Real Valladolid", "Oviedo": "Real Oviedo",
    "FC Koln": "FC Cologne", "M'gladbach": "Borussia Mönchengladbach", "Ein Frankfurt": "Eintracht Frankfurt", "Leverkusen": "Bayer Leverkusen",
    "Heidenheim": "1. FC Heidenheim 1846", "Union Berlin": "1. FC Union Berlin", "Hamburg": "Hamburg SV", "Paderborn": "SC Paderborn 07",
}


def domestic_timelines(data: Path, hist: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """match_id -> (home_max_lead, away_max_lead) from the ESPN domestic extract, matched by league, date and team name."""
    p = data / "domestic_espn.csv"
    if not p.exists():
        return {}
    e = pd.read_csv(p, dtype=str, keep_default_na=False)
    e = e[(e["timeline_ok"] == "1") & (e["home_max_lead"] != "")]
    e["date"] = pd.to_datetime(e["date_utc"], utc=True).dt.tz_convert("Europe/Vienna").dt.date
    out = {}
    by_league = {div: part for div, part in e.groupby("comp")}
    hist = hist.copy()
    hist["div"] = hist["league"].map({v: k for k, v in DIV_TO_LEAGUE.items()})
    hist["d0"] = hist["date"].dt.date
    for div, part in hist.groupby("div"):
        ev = by_league.get(div)
        if ev is None:
            continue
        names = {_norm(n): n for n in sorted(set(ev["home"]) | set(ev["away"]))}
        cache = {}

        def nm(t):
            if t not in cache:
                fixed = FD_TO_ESPN.get(t)
                cache[t] = fixed if fixed in names.values() else match_name(t, names)
            return cache[t]

        key = {(r.date, r.home, r.away): (float(r.home_max_lead), float(r.away_max_lead)) for r in ev.itertuples(index=False)}
        for r in part.itertuples(index=False):
            h, a = nm(r.home), nm(r.away)
            if h is None or a is None:
                continue
            for delta in (0, 1, -1):
                k = (r.d0 + pd.Timedelta(days=delta), h, a)
                if k in key:
                    out[r.match_id] = key[k]
                    break
    return out




def attach_timelines(data: Path, hist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """hist with home_max_lead / away_max_lead filled from the ESPN extract (NaN where unknown); returns (frame, matched)."""
    leads = domestic_timelines(data, hist)
    out = hist.copy()
    if "home_max_lead" not in out.columns:
        out["home_max_lead"] = np.nan
        out["away_max_lead"] = np.nan
    hm = out["match_id"].map(lambda m: leads.get(m, (np.nan, np.nan))[0])
    am = out["match_id"].map(lambda m: leads.get(m, (np.nan, np.nan))[1])
    out["home_max_lead"] = out["home_max_lead"].where(out["home_max_lead"].notna(), hm)
    out["away_max_lead"] = out["away_max_lead"].where(out["away_max_lead"].notna(), am)
    return out, len(leads)
