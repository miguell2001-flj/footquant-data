"""UEFA competitions for the live run: results and fixtures from the ESPN scoreboard
extract (uefa_espn.csv, fetched through Chrome), bet365 prices from The Odds API
files (odds_*.json, fetched by footquant_download.ps1 once the key is set).

Team names are canonicalised to the football-data.co.uk spelling for the big-5
clubs so the domestic history and the European matches describe the same team;
every other club keeps its ESPN name.
"""
from __future__ import annotations

import difflib
import glob
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

COMP = {"UCL": "UEFA-Champions League", "UEL": "UEFA-Europa League", "UECL": "UEFA-Conference League"}
SPORT_TO_LEAGUE = {
    "soccer_uefa_champs_league": "UEFA-Champions League",
    "soccer_uefa_europa_league": "UEFA-Europa League",
    "soccer_uefa_europa_conference_league": "UEFA-Conference League",
}
OTHER_LEAGUE = "UEFA-other"  # pseudo-league for clubs outside the five domestic files

# ESPN display name -> football-data.co.uk name (big-5 clubs only)
ESPN_TO_FD = {
    "1. FC Heidenheim 1846": "Heidenheim", "AC Milan": "Milan", "AFC Bournemouth": "Bournemouth", "AS Monaco": "Monaco",
    "AS Roma": "Roma", "Athletic Club": "Ath Bilbao", "Atlético Madrid": "Ath Madrid", "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund", "Borussia Mönchengladbach": "M'gladbach", "Brighton & Hove Albion": "Brighton",
    "Celta Vigo": "Celta", "Deportivo Alavés": "Alaves", "Eintracht Frankfurt": "Ein Frankfurt", "Espanyol": "Espanol",
    "Hellas Verona": "Verona", "Internazionale": "Inter", "Leicester City": "Leicester", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Paris Saint-Germain": "Paris SG", "Rayo Vallecano": "Vallecano", "Real Betis": "Betis", "Real Sociedad": "Sociedad",
    "SC Freiburg": "Freiburg", "Stade Brestois": "Brest", "Stade Rennais": "Rennes", "TSG Hoffenheim": "Hoffenheim",
    "Tottenham Hotspur": "Tottenham", "Union Berlin": "Union Berlin", "VfB Stuttgart": "Stuttgart", "VfL Wolfsburg": "Wolfsburg",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
}


def canon(name: str) -> str:
    return ESPN_TO_FD.get(name, name)


def season_label(year) -> str:
    y = int(year)
    return f"{y}-{str(y + 1)[2:]}"


def american_to_decimal(a) -> float:
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(fc|cf|afc|sc|ac|as|ssc|us|rc|rcd|cd|ud|sd|sk|fk|nk|bk|if|club|de|the|1\.|1846|1899|1909|1904|tsg|vfb|vfl|sv|bsc|ogc|rb|real|athletic|sporting|olympique|stade|racing)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def load_espn(path: str | Path, tz: str = "Europe/Vienna"):
    """-> (matches, odds, fixtures) from the Chrome/ESPN extract."""
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date_utc"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    d["league"] = d["comp"].map(COMP)
    d["season"] = d["season"].map(season_label)
    d["home"] = d["home"].map(canon)
    d["away"] = d["away"].map(canon)
    d["match_id"] = d["league"] + "|" + d["season"] + "|" + d["date"].dt.strftime("%Y-%m-%d") + "|" + d["home"] + "|" + d["away"]
    d["neutral"] = d["neutral"].fillna(0).astype(int).astype(bool)
    played = d[d["status"] == "STATUS_FULL_TIME"].copy()  # ties decided after extra time are left out (90-minute score unknown)
    ok = played["timeline_ok"].fillna(0).astype(int) == 1
    matches = pd.DataFrame({
        "match_id": played["match_id"], "date": played["date"], "league": played["league"], "season": played["season"],
        "home": played["home"], "away": played["away"], "home_goals": played["hs"].astype(int), "away_goals": played["as"].astype(int),
        "home_xg": np.nan, "away_xg": np.nan, "neutral": played["neutral"],
        "home_max_lead": np.where(ok, played["home_max_lead"], np.nan), "away_max_lead": np.where(ok, played["away_max_lead"], np.nan),
    })
    rows = []
    priced = d[d["ml_h"].notna() & d["ml_d"].notna() & d["ml_a"].notna()]
    for r in priced.itertuples(index=False):
        closing = r.status != "STATUS_SCHEDULED"
        for sel, a in (("home", r.ml_h), ("draw", r.ml_d), ("away", r.ml_a)):
            rows.append({"match_id": r.match_id, "bookmaker": str(r.odds_provider), "market": "1X2", "selection": sel,
                         "odds": round(american_to_decimal(a), 3), "timestamp": r.date, "is_closing": closing})
    odds = pd.DataFrame(rows, columns=["match_id", "bookmaker", "market", "selection", "odds", "timestamp", "is_closing"])
    fixtures = d[d["status"] == "STATUS_SCHEDULED"][["match_id", "date", "league", "season", "home", "away", "neutral"]].copy()
    return matches, odds, fixtures


def load_odds_api(folder: str | Path, fixtures: pd.DataFrame, bookmaker: str = "bet365", tz: str = "Europe/Vienna"):
    """bet365 1X2 prices for the ESPN fixtures from odds_<sport>.json files. -> (odds frame, log lines)"""
    rows, log = [], []
    fx = fixtures.copy()
    fx["day"] = fx["date"].dt.strftime("%Y-%m-%d")
    for f in sorted(glob.glob(str(Path(folder) / "odds_*.json"))):
        sport = Path(f).stem[len("odds_"):]
        league = SPORT_TO_LEAGUE.get(sport)
        if league is None:
            continue
        try:
            events = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.append(f"{Path(f).name}: unreadable ({e})")
            continue
        if not isinstance(events, list):
            log.append(f"{Path(f).name}: not an event list ({str(events)[:80]})")
            continue
        n_match, n_book = 0, 0
        for ev in events:
            t = pd.Timestamp(ev["commence_time"]).tz_convert(tz).tz_localize(None)
            cands = fx[(fx["league"] == league) & (fx["day"] == t.strftime("%Y-%m-%d"))]
            if len(cands) == 0:
                continue
            names = {_norm(x): x for x in pd.concat([cands["home"], cands["away"]])}
            def pick(n):
                k = _norm(n)
                if k in names:
                    return names[k]
                c = difflib.get_close_matches(k, list(names), n=1, cutoff=0.6)
                return names[c[0]] if c else None
            h, a = pick(ev["home_team"]), pick(ev["away_team"])
            m = cands[(cands["home"] == h) & (cands["away"] == a)]
            if len(m) != 1:
                log.append(f"{league}: no fixture for {ev['home_team']} v {ev['away_team']} on {t:%d %b}")
                continue
            n_match += 1
            book = next((b for b in ev.get("bookmakers", []) if b.get("key") == bookmaker), None)
            if book is None:
                continue
            mk = next((x for x in book.get("markets", []) if x.get("key") == "h2h"), None)
            if mk is None:
                continue
            prices = {}
            for o in mk.get("outcomes", []):
                if o["name"] == ev["home_team"]:
                    prices["home"] = float(o["price"])
                elif o["name"] == ev["away_team"]:
                    prices["away"] = float(o["price"])
                elif o["name"].lower() == "draw":
                    prices["draw"] = float(o["price"])
            if len(prices) != 3:
                continue
            n_book += 1
            ts = pd.Timestamp(mk.get("last_update", ev["commence_time"])).tz_convert(tz).tz_localize(None)
            for sel, p in prices.items():
                rows.append({"match_id": m["match_id"].iloc[0], "bookmaker": bookmaker, "market": "1X2", "selection": sel,
                             "odds": p, "timestamp": ts, "is_closing": False})
        log.append(f"{Path(f).name}: {len(events)} events, {n_match} matched to fixtures, {n_book} with {bookmaker} prices")
    return pd.DataFrame(rows, columns=["match_id", "bookmaker", "market", "selection", "odds", "timestamp", "is_closing"]), log


def team_leagues(domestic: pd.DataFrame, uefa: pd.DataFrame) -> dict[str, str]:
    """team -> league for the joint fit: the domestic league where known, else the UEFA pseudo-league."""
    tl = {}
    last = domestic.sort_values("date")
    for col in ("home", "away"):
        for t, lg in zip(last[col], last["league"]):
            tl[t] = lg
    for t in pd.concat([uefa["home"], uefa["away"]]).unique():
        tl.setdefault(t, OTHER_LEAGUE)
    return tl
