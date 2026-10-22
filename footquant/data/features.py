"""Contextual features (Phase 1): home advantage, travel, congestion, absences.

Every feature is computed strictly from information available before kick-off:
congestion uses prior fixtures only, absences are the pre-match announced list,
travel is geometry. The DC home-advantage parameter and the Elo home bonus are
estimated inside their respective models; `neutral` suppresses both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, float), np.asarray(lon1, float),
                                              np.asarray(lat2, float), np.asarray(lon2, float)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def travel_distance(matches: pd.DataFrame, stadiums: pd.DataFrame) -> pd.Series:
    """Km travelled by the away side: distance between the two clubs' home grounds.

    Neutral-venue matches need a venue table; absent that they get the same value.
    Missing coordinates -> NaN (the boosting layer handles NaN natively).
    """
    coords = stadiums.set_index("team")[["lat", "lon"]]
    h = coords.reindex(matches["home"])
    a = coords.reindex(matches["away"])
    d = haversine_km(h["lat"].values, h["lon"].values, a["lat"].values, a["lon"].values)
    return pd.Series(d, index=matches.index, name="away_travel_km")


def schedule_congestion(matches: pd.DataFrame, windows: tuple[int, ...] = (7, 14)) -> pd.DataFrame:
    """Rest days and fixture counts in the preceding windows, per side.

    Returns a frame indexed like `matches` with columns
    home_rest_days, away_rest_days, home_matches_{w}d, away_matches_{w}d.
    The match table should contain every competitive fixture (league, cups,
    Europe) or congestion will be understated. Values are computed from fixtures
    dated strictly before each kick-off - no look-ahead.
    """
    long = (
        pd.concat(
            [
                matches[["match_id", "date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
                matches[["match_id", "date", "away"]].rename(columns={"away": "team"}).assign(side="away"),
            ]
        )
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )
    long["rest_days"] = long.groupby("team")["date"].diff().dt.total_seconds() / 86400.0
    for w in windows:
        counts = np.zeros(len(long))
        for _, grp in long.groupby("team", sort=False):
            dates = grp["date"].values.astype("datetime64[ns]")
            # number of this team's fixtures in [kick-off - w days, kick-off)
            lo = np.searchsorted(dates, dates - np.timedelta64(w, "D"), side="left")
            counts[grp.index.values] = np.arange(len(dates)) - lo
        long[f"matches_{w}d"] = counts
    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        part = long[long["side"] == side].set_index("match_id")
        out[f"{side}_rest_days"] = matches["match_id"].map(part["rest_days"]).values
        for w in windows:
            out[f"{side}_matches_{w}d"] = matches["match_id"].map(part[f"matches_{w}d"]).values
    return out


def absence_features(matches: pd.DataFrame, absences: pd.DataFrame | None) -> pd.DataFrame:
    """Aggregate announced absences into per-side importance sums and counts.

    `importance` should be a pre-match measure of the player's weight in the
    squad over the past year (share of minutes, or share of xG + xA). Summing
    gives "missing share of the team"; the boosted adjuster learns the non-linear
    effect on scoring and conceding rates.
    """
    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        out[f"{side}_missing_importance"] = 0.0
        out[f"{side}_missing_count"] = 0.0
    if absences is None or absences.empty:
        return out
    agg = absences.groupby(["match_id", "team"]).agg(imp=("importance", "sum"), n=("player", "size"))
    for side in ("home", "away"):
        keys = list(zip(matches["match_id"], matches[side]))
        out[f"{side}_missing_importance"] = [float(agg["imp"].get(k, 0.0)) for k in keys]
        out[f"{side}_missing_count"] = [float(agg["n"].get(k, 0.0)) for k in keys]
    return out


def max_lead_from_shots(shots: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Largest lead each side held at any point, from goal events in the shot table.

    Needs columns match_id, team, minute, is_goal. Returns match_id,
    home_max_lead, away_max_lead (0 when the side never led). This is what
    settles bet365's early payout in the backtest.
    """
    goals = shots[shots["is_goal"].astype(bool)].copy()
    goals["minute"] = goals["minute"].astype(float)
    side = {}
    for mid, h, a in zip(matches["match_id"], matches["home"], matches["away"]):
        side[(mid, h)] = 1
        side[(mid, a)] = -1
    goals["step"] = [side.get((m, t), 0) for m, t in zip(goals["match_id"], goals["team"])]
    goals = goals.sort_values(["match_id", "minute"])
    goals["diff"] = goals.groupby("match_id")["step"].cumsum()
    lead = goals.groupby("match_id")["diff"].agg(home_max_lead=lambda s: max(s.max(), 0),
                                                 away_max_lead=lambda s: max(-s.min(), 0))
    out = matches[["match_id"]].merge(lead, on="match_id", how="left").fillna(0)
    out["home_max_lead"] = out["home_max_lead"].astype(int)
    out["away_max_lead"] = out["away_max_lead"].astype(int)
    return out


def build_feature_frame(
    matches: pd.DataFrame,
    stadiums: pd.DataFrame | None = None,
    absences: pd.DataFrame | None = None,
    elo_pre: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the contextual feature frame used by the boosted rate adjuster.

    Columns (all pre-match):
      neutral, away_travel_km, home/away_rest_days, home/away_matches_7d/14d,
      home/away_missing_importance, home/away_missing_count,
      elo_home_pre, elo_away_pre, elo_diff (from `LeagueElo.run`).
    """
    feats = pd.DataFrame(index=matches.index)
    feats["match_id"] = matches["match_id"].values
    feats["neutral"] = matches["neutral"].astype(float).values if "neutral" in matches.columns else 0.0
    feats["away_travel_km"] = travel_distance(matches, stadiums).values if stadiums is not None else np.nan
    feats = pd.concat([feats, schedule_congestion(matches), absence_features(matches, absences)], axis=1)
    if elo_pre is not None:
        e = elo_pre.set_index("match_id")
        for c in ("elo_home_pre", "elo_away_pre", "elo_diff"):
            feats[c] = matches["match_id"].map(e[c]).values
    return feats
