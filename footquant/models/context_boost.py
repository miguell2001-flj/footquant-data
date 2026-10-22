"""Boosted contextual rate adjuster (Phase 3).

Architecture
------------
Dixon-Coles supplies the baseline scoring rates (lambda, mu). For data-rich
leagues a gradient-boosted Poisson model learns a multiplicative correction
from contextual, non-linear covariates:

    lambda_final = lambda_DC * exp(f(x))       (same for mu with the away row)

f is fitted with a Poisson objective and the DC rate as an *offset*
(log lambda_DC enters as base_margin in XGBoost). Because the output is still a
pair of rates, the full score matrix - and with it 1X2, BTTS and O/U 2.5 -
comes from the same `score_matrix` mapping; the ML layer never produces market
probabilities of its own that would have to be re-coupled to a score grid.

Each match contributes two training rows (home perspective, away perspective)
with features: elo_diff (own minus opponent), rest days and fixture counts for
both sides, travel, missing-player importance for both sides, is_home. The
training base rate for a match must come from a fit that did not see that
match - the walk-forward backtest produces exactly those out-of-sample rates,
which is why `BettingPipeline.fit_context_layer` consumes backtest output.

Backends: XGBoost (`count:poisson`, offset via base_margin) when installed,
otherwise scikit-learn's HistGradientBoostingRegressor with Poisson loss, using
the exact equivalence "Poisson regression of y with offset o" = "Poisson
regression of y/o with sample weight o".

`shrink` in (0, 1] pulls the learned factor toward 1 (factor ** shrink) - use
0.5-0.7 for mid-sized leagues and 0 (i.e. skip the layer) for sparse ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # optional dependency
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:  # pragma: no cover - exercised only without xgboost
    _HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingRegressor

FEATURE_COLUMNS: list[str] = [
    "is_home",
    "elo_diff",
    "own_rest_days",
    "opp_rest_days",
    "own_matches_7d",
    "opp_matches_7d",
    "own_matches_14d",
    "opp_matches_14d",
    "own_travel_km",
    "opp_travel_km",
    "own_missing_importance",
    "opp_missing_importance",
    "own_missing_count",
    "opp_missing_count",
]


def team_rows_from_predictions(preds: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Reshape match-level predictions + features into one row per (match, side).

    `preds` needs match_id, lam, mu, home_goals, away_goals (and optionally
    home_xg, away_xg); `feats` is `build_feature_frame` output.
    Returns the long frame with FEATURE_COLUMNS, y_goals, y_xg, base_rate, side.
    """
    f = feats.set_index("match_id")
    rows = []
    for rec in preds.to_dict("records"):
        fx = f.loc[rec["match_id"]] if rec["match_id"] in f.index else {}
        g = lambda k, default=np.nan: float(fx.get(k, default)) if len(fx) else default
        common = {"match_id": rec["match_id"]}
        home = {
            **common,
            "side": "home",
            "is_home": 1.0 - g("neutral", 0.0),
            "elo_diff": g("elo_diff"),
            "own_rest_days": g("home_rest_days"),
            "opp_rest_days": g("away_rest_days"),
            "own_matches_7d": g("home_matches_7d"),
            "opp_matches_7d": g("away_matches_7d"),
            "own_matches_14d": g("home_matches_14d"),
            "opp_matches_14d": g("away_matches_14d"),
            "own_travel_km": 0.0,
            "opp_travel_km": g("away_travel_km"),
            "own_missing_importance": g("home_missing_importance", 0.0),
            "opp_missing_importance": g("away_missing_importance", 0.0),
            "own_missing_count": g("home_missing_count", 0.0),
            "opp_missing_count": g("away_missing_count", 0.0),
            "base_rate": float(rec["lam"]),
            "y_goals": float(rec.get("home_goals", np.nan)),
            "y_xg": float(rec.get("home_xg", np.nan)),
        }
        away = {
            **common,
            "side": "away",
            "is_home": 0.0,
            "elo_diff": -g("elo_diff"),
            "own_rest_days": g("away_rest_days"),
            "opp_rest_days": g("home_rest_days"),
            "own_matches_7d": g("away_matches_7d"),
            "opp_matches_7d": g("home_matches_7d"),
            "own_matches_14d": g("away_matches_14d"),
            "opp_matches_14d": g("home_matches_14d"),
            "own_travel_km": g("away_travel_km"),
            "opp_travel_km": 0.0,
            "own_missing_importance": g("away_missing_importance", 0.0),
            "opp_missing_importance": g("home_missing_importance", 0.0),
            "own_missing_count": g("away_missing_count", 0.0),
            "opp_missing_count": g("home_missing_count", 0.0),
            "base_rate": float(rec["mu"]),
            "y_goals": float(rec.get("away_goals", np.nan)),
            "y_xg": float(rec.get("away_xg", np.nan)),
        }
        rows.extend([home, away])
    return pd.DataFrame(rows)


class ContextualRateAdjuster:
    def __init__(
        self,
        backend: str = "auto",
        shrink: float = 1.0,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 3,
        min_child_weight: float = 25.0,
        seed: int = 0,
        features: list[str] | None = None,
    ) -> None:
        if backend == "auto":
            backend = "xgboost" if _HAS_XGB else "sklearn"
        if backend == "xgboost" and not _HAS_XGB:
            raise ImportError("xgboost is not installed; use backend='sklearn'")
        if not 0.0 <= shrink <= 1.0:
            raise ValueError("shrink must be in [0, 1]")
        self.backend = backend
        self.shrink = shrink
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.seed = seed
        self.features = features or FEATURE_COLUMNS
        self._model = None

    def _X(self, rows: pd.DataFrame) -> np.ndarray:
        return rows.reindex(columns=self.features).astype(float).values

    def fit(self, rows: pd.DataFrame, target: str = "y_goals") -> "ContextualRateAdjuster":
        """Fit f(x) with Poisson loss and offset log(base_rate). `target`: y_goals or y_xg."""
        rows = rows.dropna(subset=[target, "base_rate"])
        X = self._X(rows)
        y = rows[target].values.astype(float)
        base = rows["base_rate"].values.astype(float)
        if self.backend == "xgboost":
            dtrain = xgb.DMatrix(X, label=y, base_margin=np.log(base), feature_names=self.features)
            params = {
                "objective": "count:poisson",
                "eta": self.learning_rate,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda": 1.0,
                "max_delta_step": 0.7,
                "base_score": 1.0,
                "seed": self.seed,
                "verbosity": 0,
            }
            self._model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        else:
            est = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=int(self.min_child_weight),
                random_state=self.seed,
            )
            est.fit(X, y / base, sample_weight=base)
            self._model = est
        return self

    def predict_factor(self, rows: pd.DataFrame) -> np.ndarray:
        """Multiplicative factor exp(f(x)) ** shrink applied to the DC rate."""
        if self._model is None:
            raise RuntimeError("adjuster is not fitted")
        X = self._X(rows)
        if self.backend == "xgboost":
            d = xgb.DMatrix(X, base_margin=np.zeros(len(X)), feature_names=self.features)
            factor = np.exp(self._model.predict(d, output_margin=True))
        else:
            factor = self._model.predict(X)
        return np.clip(factor, 0.25, 4.0) ** self.shrink

    def adjust(self, preds: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        """Return `preds` with lam/mu multiplied by the learned factors (columns
        lam_adj, mu_adj, home_factor, away_factor)."""
        long = team_rows_from_predictions(preds, feats)
        long["factor"] = self.predict_factor(long)
        wide = long.pivot(index="match_id", columns="side", values="factor")
        out = preds.copy()
        out["home_factor"] = out["match_id"].map(wide["home"]).values
        out["away_factor"] = out["match_id"].map(wide["away"]).values
        out["lam_adj"] = out["lam"] * out["home_factor"]
        out["mu_adj"] = out["mu"] * out["away_factor"]
        return out
