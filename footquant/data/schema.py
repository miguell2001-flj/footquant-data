"""Canonical data structures.

Everything downstream consumes plain pandas DataFrames with the column sets
defined here. Keeping the contract explicit makes the look-ahead rules in the
backtest auditable: a match row must contain nothing that was unknown at kick-off
except the realised goals / xG, which are only ever read for matches dated
strictly before the prediction time.

Match table (one row per match, all competitions the modelled teams play in):
    match_id      str      unique id
    date          datetime kick-off (tz-naive, local or UTC - be consistent)
    league        str      competition code, e.g. "ENG-Premier League"
    season        str      e.g. "2025-26"
    home, away    str      team names (canonical - apply alias maps on ingest)
    home_goals    int
    away_goals    int
    home_xg       float    match-level xG (sum of shot xG), NaN if unavailable
    away_xg       float
  optional:
    neutral       bool     neutral venue (no home advantage in features)
    home_max_lead int      largest lead the home side held at any point (from
    away_max_lead int      goal timeline) - needed to settle early payouts

Shot table (one row per shot) - the source of match-level xG:
    match_id, team, player, minute, xg, is_goal, is_penalty, x, y

Odds table (one row per price observation):
    match_id, bookmaker, market ("1X2" | "BTTS" | "OU25"), selection
    ("home" | "draw" | "away" | "yes" | "no" | "over" | "under"), odds (decimal),
    timestamp, is_closing (bool)

Absence table: match_id, team, player, importance (0..1 share of team value,
e.g. minutes share or xG+xA share over the previous 12 months)

Stadium table: team, lat, lon
"""

from __future__ import annotations

import pandas as pd

MATCH_COLUMNS: list[str] = [
    "match_id",
    "date",
    "league",
    "season",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
]
SHOT_COLUMNS: list[str] = ["match_id", "team", "player", "minute", "xg", "is_goal", "is_penalty"]
ODDS_COLUMNS: list[str] = ["match_id", "bookmaker", "market", "selection", "odds", "timestamp", "is_closing"]
ABSENCE_COLUMNS: list[str] = ["match_id", "team", "player", "importance"]
STADIUM_COLUMNS: list[str] = ["team", "lat", "lon"]

MARKET_SELECTIONS: dict[str, tuple[str, ...]] = {
    "1X2": ("home", "draw", "away"),
    "BTTS": ("yes", "no"),
    "OU25": ("over", "under"),
}


def validate_matches(df: pd.DataFrame, require_xg: bool = False) -> pd.DataFrame:
    """Return a validated, date-sorted copy of a match table.

    Raises if required columns are missing; coerces dtypes; adds NaN xG columns
    when absent (and `require_xg` is False) so the goals-only model still runs.
    """
    required = [c for c in MATCH_COLUMNS if c not in ("home_xg", "away_xg")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"match table is missing columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ("home_goals", "away_goals"):
        out[col] = out[col].astype(int)
    for col in ("home_xg", "away_xg"):
        if col not in out.columns:
            if require_xg:
                raise ValueError(f"match table is missing {col}")
            out[col] = float("nan")
        out[col] = out[col].astype(float)
    for col in ("match_id", "league", "season", "home", "away"):
        out[col] = out[col].astype(str)
    if out["match_id"].duplicated().any():
        raise ValueError("match_id must be unique")
    if (out["home"] == out["away"]).any():
        raise ValueError("a team cannot play itself")
    return out.sort_values(["date", "match_id"]).reset_index(drop=True)


def shots_to_match_xg(shots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a shot table to one row per (match_id, team).

    Returns columns: match_id, team, xg (all shots), npxg (non-penalty xG),
    shots, goals. Non-penalty xG is the better strength signal - penalties are
    nearly 0.76-0.79 xG each and largely luck-driven.
    """
    missing = [c for c in ("match_id", "team", "xg") if c not in shots.columns]
    if missing:
        raise ValueError(f"shot table is missing columns: {missing}")
    s = shots.copy()
    if "is_penalty" not in s.columns:
        s["is_penalty"] = False
    if "is_goal" not in s.columns:
        s["is_goal"] = False
    s["npxg"] = s["xg"].where(~s["is_penalty"].astype(bool), 0.0)
    agg = (
        s.groupby(["match_id", "team"], as_index=False)
        .agg(xg=("xg", "sum"), npxg=("npxg", "sum"), shots=("xg", "size"), goals=("is_goal", "sum"))
    )
    agg["goals"] = agg["goals"].astype(int)
    return agg


def attach_xg(matches: pd.DataFrame, team_xg: pd.DataFrame, use: str = "npxg") -> pd.DataFrame:
    """Fill home_xg / away_xg on the match table from `shots_to_match_xg` output."""
    if use not in ("xg", "npxg"):
        raise ValueError("use must be 'xg' or 'npxg'")
    out = matches.copy()
    lookup = team_xg.set_index(["match_id", "team"])[use]
    out["home_xg"] = [lookup.get((m, t), float("nan")) for m, t in zip(out["match_id"], out["home"])]
    out["away_xg"] = [lookup.get((m, t), float("nan")) for m, t in zip(out["match_id"], out["away"])]
    return out
