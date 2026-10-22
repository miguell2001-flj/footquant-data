"""Phase 1 - data acquisition and feature engineering."""

from .elo import LeagueElo
from .features import (
    absence_features,
    build_feature_frame,
    haversine_km,
    max_lead_from_shots,
    schedule_congestion,
    travel_distance,
)
from .ingest import (
    FD_TO_UNDERSTAT_EPL,
    load_football_data_csv,
    load_understat_matches,
    load_understat_shots,
    merge_xg,
    read_football_data_csv,
)
from .schema import (
    ABSENCE_COLUMNS,
    MATCH_COLUMNS,
    ODDS_COLUMNS,
    SHOT_COLUMNS,
    STADIUM_COLUMNS,
    attach_xg,
    shots_to_match_xg,
    validate_matches,
)

__all__ = [
    "ABSENCE_COLUMNS",
    "FD_TO_UNDERSTAT_EPL",
    "load_understat_shots",
    "read_football_data_csv",
    "LeagueElo",
    "MATCH_COLUMNS",
    "ODDS_COLUMNS",
    "SHOT_COLUMNS",
    "STADIUM_COLUMNS",
    "absence_features",
    "attach_xg",
    "build_feature_frame",
    "haversine_km",
    "load_football_data_csv",
    "load_understat_matches",
    "max_lead_from_shots",
    "merge_xg",
    "schedule_congestion",
    "shots_to_match_xg",
    "travel_distance",
    "validate_matches",
]
