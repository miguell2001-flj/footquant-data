"""Settlement checks for slips (run: python3 tests/test_slips.py). Legs are stored the way the Desk holds them:
short league name in `league`, full code in `league_code`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from footquant.market.exchange import settle_slip  # noqa: E402
from footquant.config import DEFAULT_EARLY_PAYOUT_COMPETITIONS as EP  # noqa: E402

LEGS = [{"match_id": "a", "league": "Ligue 1", "league_code": "FRA-Ligue 1", "selection": "home", "odds": 1.45},
        {"match_id": "b", "league": "Premier League", "league_code": "ENG-Premier League", "selection": "away", "odds": 1.65}]
SLIP = {"legs": LEGS, "stake": 78.0, "odds": 2.3925}
CASES = [
    ("both win", {"a": (2, 0, 2.0, 0.0), "b": (0, 1, 0.0, 1.0)}, "won"),
    ("a leg draws, never led by two", {"a": (1, 1, 1.0, 0.0), "b": (0, 1, 0.0, 1.0)}, "lost"),
    ("a leg drew after leading 2-0 (early payout)", {"a": (2, 2, 2.0, 0.0), "b": (0, 1, 0.0, 1.0)}, "won"),
    ("away leg led 2-0 and lost (early payout)", {"a": (2, 0, 2.0, 0.0), "b": (3, 2, 1.0, 2.0)}, "won"),
    ("a leg not played yet", {"a": (2, 0, 2.0, 0.0)}, None),
    ("no goal timeline, could have led by two", {"a": (2, 2, None, None), "b": (0, 1, 0.0, 1.0)}, None),
    ("no goal timeline, never scored two", {"a": (1, 1, None, None), "b": (0, 1, 0.0, 1.0)}, "lost"),
]
fails = 0
for name, res, exp in CASES:
    o = settle_slip(SLIP, res, EP)
    got = o["status"] if o else None
    fails += got != exp
    print(("PASS " if got == exp else "FAIL ") + name, got)
outside = {"legs": [{"match_id": "c", "league": "Other", "league_code": "XXX", "selection": "home", "odds": 1.5}], "stake": 10.0, "odds": 1.5}
o = settle_slip(outside, {"c": (2, 2, 2.0, 0.0)}, EP)
fails += o["status"] != "lost"
print(("PASS " if o["status"] == "lost" else "FAIL ") + "no early payout outside the eligible competitions")
sys.exit(1 if fails else 0)
