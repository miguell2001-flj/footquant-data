"""Comeback record and early-payout curve checks (run: python3 tests/test_comeback.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from footquant.market.exchange import EpCurve, comeback_rates  # noqa: E402

# A trails 0-2 and draws (comeback), A trails 0-2 and loses, B never trails by two.
H = pd.DataFrame([
    {"match_id": "m1", "date": "2026-08-01", "home": "A", "away": "B", "home_goals": 2, "away_goals": 2, "home_max_lead": 0, "away_max_lead": 2},
    {"match_id": "m2", "date": "2026-08-08", "home": "C", "away": "A", "home_goals": 3, "away_goals": 1, "home_max_lead": 2, "away_max_lead": 0},
    {"match_id": "m3", "date": "2026-08-15", "home": "A", "away": "C", "home_goals": 1, "away_goals": 0, "home_max_lead": 1, "away_max_lead": 0},
])
prior, now = comeback_rates(H)
checks = [
    ("prior rate uses only earlier matches", prior[("m1", "A")] == 0.1),
    ("after one comeback in one", abs(prior[("m2", "A")] - 2 / 11) < 1e-12),
    ("after one comeback in two", abs(prior[("m3", "A")] - 2 / 12) < 1e-12),
    ("current rate counts every match", abs(now["A"] - 2 / 12) < 1e-12),
    ("team that never trailed by two stays at the prior", now["B"] == 0.1 and now["C"] == 0.1),
]
c = EpCurve(coef=np.array([-3.86, -0.06, -0.019, 0.27]), n=1, z_mu=0.1, z_sd=0.05)
checks += [
    ("unknown opponent gives z = 0", c.zscore(None) == 0.0 and c.zscore(float("nan")) == 0.0),
    ("z capped at +2 and -2", c.zscore(0.9) == 2.0 and c.zscore(0.0) == -2.0),
    ("strong comeback opponent raises EP", c(0.6, 1.0) > c(0.6, 0.0) > c(0.6, -1.0)),
    ("curve ignores z beyond the cap", c(0.6, 5.0) == c(0.6, 2.0)),
]
fails = 0
for name, ok in checks:
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(1 if fails else 0)
