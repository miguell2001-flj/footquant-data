"""Loaders for the two free sources that cover the bulk of the data requirement.

* football-data.co.uk  - results plus bet365 opening (B365H/D/A) and closing
  (B365CH/CD/CA) 1X2 odds and bet365 over/under 2.5 prices for every major
  European league back to the 1990s. This is the backtest odds source.
* Understat            - match-level and shot-level xG (2014/15 onwards) for the
  big-5 leagues plus the Russian Premier League. Shot JSON includes minute and
  result, so goal timelines (early-payout settlement) can be derived.

Team names differ between sources; pass an alias map ({"Man United": "Manchester United"})
so that every table uses one canonical name per club.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import MATCH_COLUMNS, validate_matches

_FD_ODDS_COLUMNS: dict[str, tuple[str, str, bool]] = {
    # column -> (market, selection, is_closing)
    "B365H": ("1X2", "home", False),
    "B365D": ("1X2", "draw", False),
    "B365A": ("1X2", "away", False),
    "B365CH": ("1X2", "home", True),
    "B365CD": ("1X2", "draw", True),
    "B365CA": ("1X2", "away", True),
    "B365>2.5": ("OU25", "over", False),
    "B365<2.5": ("OU25", "under", False),
    "B365C>2.5": ("OU25", "over", True),
    "B365C<2.5": ("OU25", "under", True),
}


def _canon(name: str, aliases: dict[str, str] | None) -> str:
    name = str(name).strip()
    return aliases.get(name, name) if aliases else name


# football-data.co.uk -> Understat spellings for Premier League clubs
FD_TO_UNDERSTAT_EPL: dict[str, str] = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
}


def read_football_data_csv(path: str | Path) -> pd.DataFrame:
    """Tolerant reader for football-data.co.uk CSVs (older files carry ragged
    trailing commas and two-digit years)."""
    import io

    raw_bytes = Path(path).read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")  # current files are UTF-8 with a BOM
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")  # older files
    lines = text.lstrip("\ufeff").splitlines()
    ncol = len(lines[0].split(","))
    fixed = [",".join(line.split(",")[:ncol]) for line in lines if line.strip(",").strip()]
    return pd.read_csv(io.StringIO("\n".join(fixed)))


def load_football_data_csv(
    path: str | Path,
    league: str,
    season: str,
    aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one football-data.co.uk season file.

    Returns (matches, odds). xG columns are NaN - join Understat with `merge_xg`.
    `match_id` is "<league>|<season>|<date>|<home>|<away>".
    """
    raw = read_football_data_csv(path)
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    date = pd.to_datetime(raw["Date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    # the 2026/27 files carry match xG; older files do not
    xg_cols = next((c for c in (("HxG", "AxG"), ("HXG", "AXG"), ("Home xG", "Away xG")) if c[0] in raw.columns and c[1] in raw.columns), None)
    if "Time" in raw.columns:
        time = pd.to_timedelta(raw["Time"].fillna("15:00").astype(str) + ":00", errors="coerce")
        date = date + time.fillna(pd.Timedelta(hours=15))
    home = raw["HomeTeam"].map(lambda t: _canon(t, aliases))
    away = raw["AwayTeam"].map(lambda t: _canon(t, aliases))
    match_id = [
        f"{league}|{season}|{d.date()}|{h}|{a}" for d, h, a in zip(date, home, away)
    ]
    matches = pd.DataFrame(
        {
            "match_id": match_id,
            "date": date,
            "league": league,
            "season": season,
            "home": home,
            "away": away,
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
            "home_xg": pd.to_numeric(raw[xg_cols[0]], errors="coerce").values if xg_cols else float("nan"),
            "away_xg": pd.to_numeric(raw[xg_cols[1]], errors="coerce").values if xg_cols else float("nan"),
        }
    )
    if "HTHG" in raw.columns and "HTAG" in raw.columns:
        matches["ht_home_goals"] = pd.to_numeric(raw["HTHG"], errors="coerce").values
        matches["ht_away_goals"] = pd.to_numeric(raw["HTAG"], errors="coerce").values
    rows = []
    for col, (market, selection, closing) in _FD_ODDS_COLUMNS.items():
        if col not in raw.columns:
            continue
        for mid, d, price in zip(match_id, date, raw[col]):
            if pd.isna(price) or price <= 1.0:
                continue
            rows.append(
                {
                    "match_id": mid,
                    "bookmaker": "bet365",
                    "market": market,
                    "selection": selection,
                    "odds": float(price),
                    "timestamp": d,
                    "is_closing": closing,
                }
            )
    odds = pd.DataFrame(rows)
    return validate_matches(matches), odds


DIV_TO_LEAGUE: dict[str, str] = {
    "E0": "ENG-Premier League",
    "E1": "ENG-Championship",
    "SP1": "ESP-La Liga",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "F1": "FRA-Ligue 1",
}


def load_football_data_fixtures(
    path: str | Path,
    season: str,
    aliases: dict[str, str] | None = None,
    divisions: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read football-data.co.uk `fixtures.csv` (upcoming matches with current prices).

    Returns (fixtures, odds): fixtures has match_id, date, league, home, away;
    odds is an ODDS_COLUMNS frame (is_closing=False). match_id follows the same
    "<league>|<season>|<date>|<home>|<away>" convention as `load_football_data_csv`,
    so the bets settle against the season file once the results are in.
    """
    divisions = divisions or DIV_TO_LEAGUE
    raw = read_football_data_csv(path)
    raw = raw.dropna(subset=["Div", "HomeTeam", "AwayTeam"])
    raw = raw[raw["Div"].isin(divisions)]
    date = pd.to_datetime(raw["Date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    if "Time" in raw.columns:
        time = pd.to_timedelta(raw["Time"].fillna("15:00").astype(str) + ":00", errors="coerce")
        date = date + time.fillna(pd.Timedelta(hours=15))
    league = raw["Div"].map(divisions)
    home = raw["HomeTeam"].map(lambda t: _canon(t, aliases))
    away = raw["AwayTeam"].map(lambda t: _canon(t, aliases))
    match_id = [f"{lg}|{season}|{d.date()}|{h}|{a}" for lg, d, h, a in zip(league, date, home, away)]
    fixtures = pd.DataFrame({"match_id": match_id, "date": date, "league": league.values, "home": home.values, "away": away.values})
    rows = []
    for col, (market, selection, closing) in _FD_ODDS_COLUMNS.items():
        if col not in raw.columns or closing:
            continue
        for mid, d, price in zip(match_id, date, raw[col]):
            if pd.isna(price) or price <= 1.0:
                continue
            rows.append({"match_id": mid, "bookmaker": "bet365", "market": market, "selection": selection,
                         "odds": float(price), "timestamp": d, "is_closing": False})
    return fixtures.reset_index(drop=True), pd.DataFrame(rows)


def load_understat_matches(
    source: str | Path | Iterable[dict],
    league: str,
    season: str,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Parse Understat league match JSON (the `datesData` payload or a saved copy).

    Each record looks like {"id": "...", "h": {"title": ...}, "a": {...},
    "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.83", "a": "0.74"},
    "datetime": "2025-08-16 15:00:00", "isResult": true}.
    """
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as fh:
            records = json.load(fh)
    else:
        records = list(source)
    rows = []
    for r in records:
        if not r.get("isResult", True):
            continue
        rows.append(
            {
                "match_id": f"understat|{r['id']}",
                "date": pd.to_datetime(r["datetime"]),
                "league": league,
                "season": season,
                "home": _canon(r["h"]["title"], aliases),
                "away": _canon(r["a"]["title"], aliases),
                "home_goals": int(r["goals"]["h"]),
                "away_goals": int(r["goals"]["a"]),
                "home_xg": float(r["xG"]["h"]),
                "away_xg": float(r["xG"]["a"]),
            }
        )
    return validate_matches(pd.DataFrame(rows))


def load_understat_shots(
    source: str | Path | pd.DataFrame,
    league: str,
    aliases: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a match table and a canonical shot table from Understat shot data.

    Accepts the worldfootballR `understat_shots` release format (columns id,
    minute, result, X, Y, xG, player, h_a / home_away, situation, season,
    match_id, home_team, away_team, home_goals, away_goals, date) as a CSV
    path or DataFrame. Returns (matches, shots):

      matches: MATCH_COLUMNS + home_max_lead, away_max_lead (from the goal
               timeline, own goals credited to the opponent; NaN for the rare
               match whose events disagree with the recorded score unless
               `strict=True`, which raises), home_npxg, away_npxg
      shots:   SHOT_COLUMNS with `team` as the shooting side's club name
    """
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, low_memory=False)
    side = df["h_a"] if "h_a" in df.columns else pd.Series(index=df.index, dtype=object)
    if "home_away" in df.columns:
        side = side.fillna(df["home_away"])
    df["side"] = side
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].map(lambda t: _canon(t, aliases))
    df["away_team"] = df["away_team"].map(lambda t: _canon(t, aliases))
    df["team"] = np.where(df["side"] == "h", df["home_team"], df["away_team"])
    df["is_goal"] = df["result"].isin(["Goal", "OwnGoal"])
    df["is_penalty"] = df["situation"].eq("Penalty")
    df["match_id"] = "understat|" + df["match_id"].astype(int).astype(str)
    # own goals: the shot is logged against the player's own side; credit the opponent, no xG
    own = df["result"].eq("OwnGoal")
    df.loc[own, "team"] = np.where(df.loc[own, "side"] == "h", df.loc[own, "away_team"], df.loc[own, "home_team"])
    df.loc[own, "xG"] = 0.0

    meta = df.groupby("match_id").agg(
        date=("date", "first"),
        home=("home_team", "first"),
        away=("away_team", "first"),
        home_goals=("home_goals", "first"),
        away_goals=("away_goals", "first"),
        season_start=("season", "first"),
    )
    df["xg_np"] = df["xG"].where(~df["is_penalty"], 0.0)
    xg = df[~own].groupby(["match_id", "side"]).agg(xg=("xG", "sum"), npxg=("xg_np", "sum"))
    for s, pre in (("h", "home"), ("a", "away")):
        part = xg.xs(s, level="side")
        meta[f"{pre}_xg"] = meta.index.map(part["xg"]).astype(float)
        meta[f"{pre}_npxg"] = meta.index.map(part["npxg"]).astype(float)
    goals = df[df["is_goal"]].sort_values(["match_id", "minute"])
    step = np.where(goals["team"] == goals["match_id"].map(meta["home"]), 1, -1)
    goals = goals.assign(step=step)
    goals["diff"] = goals.groupby("match_id")["step"].cumsum()
    lead = goals.groupby("match_id")["diff"].agg(home_max_lead=lambda s: max(int(s.max()), 0), away_max_lead=lambda s: max(int(-s.min()), 0))
    meta = meta.join(lead).fillna({"home_max_lead": 0, "away_max_lead": 0, "home_xg": 0.0, "away_xg": 0.0, "home_npxg": 0.0, "away_npxg": 0.0})
    meta["home_max_lead"] = meta["home_max_lead"].astype(int)
    meta["away_max_lead"] = meta["away_max_lead"].astype(int)
    meta["league"] = league
    meta["season"] = meta["season_start"].astype(int).map(lambda y: f"{y}-{str(y + 1)[-2:]}")
    meta = meta.drop(columns="season_start").reset_index()
    matches = validate_matches(meta[MATCH_COLUMNS + ["home_npxg", "away_npxg", "home_max_lead", "away_max_lead"]])
    # sanity: the goal timeline must reproduce the recorded score
    counts = goals.groupby("match_id")["step"].agg(home=lambda s: int((s == 1).sum()), away=lambda s: int((s == -1).sum()))
    counts = counts.reindex(matches["match_id"]).fillna(0)
    bad = (counts["home"].values != matches["home_goals"].values) | (counts["away"].values != matches["away_goals"].values)
    if bad.any():
        if strict:
            raise ValueError(f"goal events do not match recorded scores for {int(bad.sum())} matches")
        import warnings

        warnings.warn(f"{int(bad.sum())} matches have goal events that do not match the recorded score; their max-lead columns are set to NaN")
        matches.loc[bad, ["home_max_lead", "away_max_lead"]] = np.nan
    shots = df[["match_id", "team", "player", "minute", "xG", "is_goal", "is_penalty", "X", "Y", "result", "situation"]].rename(
        columns={"xG": "xg", "X": "x", "Y": "y"}
    )
    return matches, shots


def merge_xg(matches: pd.DataFrame, xg_matches: pd.DataFrame, tolerance_days: int = 1) -> pd.DataFrame:
    """Join xG (and max-lead columns if present) from one match table onto another
    by (home, away) and kick-off date within `tolerance_days`.

    Both tables must use canonical team names. Rows without a match keep NaN xG.
    """
    out = matches.copy()
    extra = [c for c in ("home_xg", "away_xg", "home_max_lead", "away_max_lead") if c in xg_matches.columns]
    right = xg_matches[["home", "away", "date"] + extra].rename(
        columns={"date": "_xg_date", **{c: f"_src_{c}" for c in extra}}
    )
    m = out.merge(right, on=["home", "away"], how="left")
    gap = (m["date"] - m["_xg_date"]).abs()
    m["_ok"] = gap <= pd.Timedelta(days=tolerance_days)
    m["_gap"] = gap.fillna(pd.Timedelta(days=36500))
    # keep the closest candidate per match, preferring ones inside the tolerance
    m = m.sort_values(["match_id", "_ok", "_gap"], ascending=[True, False, True]).drop_duplicates("match_id", keep="first")
    for c in extra:
        src = f"_src_{c}"
        vals = m[src].where(m["_ok"])
        if c in out.columns:
            vals = vals.where(m["_ok"], m[c])
        m[c] = vals
        m = m.drop(columns=src)
    m = m.drop(columns=["_xg_date", "_ok", "_gap"])
    return validate_matches(m)
