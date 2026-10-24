"""Print how many big-5 fixtures in data/fixtures.csv kick off within the next N minutes (default 80). Standard library
only, so the workflow can ask before installing anything."""
import csv
import datetime as dt
import sys
from zoneinfo import ZoneInfo

mins = int(sys.argv[1]) if len(sys.argv) > 1 else 80
now = dt.datetime.fromisoformat(sys.argv[2].replace("Z", "+00:00")) if len(sys.argv) > 2 else dt.datetime.now(dt.timezone.utc)
due = 0
with open("data/fixtures.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("Div") in ("E0", "SP1", "I1", "D1", "F1") and r.get("Date"):
            d = dt.datetime.strptime(r["Date"].strip(), "%d/%m/%Y")
            hh, mm = (r.get("Time") or "15:00").split(":")[:2]
            ko = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("Europe/London")).astimezone(dt.timezone.utc)
            due += now < ko <= now + dt.timedelta(minutes=mins)
print(due)
