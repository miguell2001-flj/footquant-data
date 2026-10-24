"""Daily data fetch for footquant, run by GitHub Actions (see .github/workflows/fetch.yml).

Writes into data/:
  E0_2627.csv, SP1_2627.csv, I1_2627.csv, D1_2627.csv, F1_2627.csv, fixtures.csv   football-data.co.uk (results, bet365 prices, xG)
  uefa_espn.csv                                                                     Champions / Europa / Conference League results,
                                                                                    goal timelines and fixtures (ESPN scoreboard)
  domestic_espn.csv                                                                 the five leagues' results and goal timelines
                                                                                    (ESPN scoreboard) for the early-payout check:
                                                                                    largest lead per side, the goal sequence with
                                                                                    minutes ("12H;34A") and the minute each side
                                                                                    first led by two
  odds_<sport>.json                                                                 bet365 prices for the European fixtures
                                                                                    (The Odds API; only when the ODDS_KEY secret is set
                                                                                    and a European match is within the next 4 days)
  fetch.log                                                                         one line per file per run
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import pathlib
import sys
import time

import requests

OUT = pathlib.Path("data")
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (footquant data fetch; github actions)"}
LOG = []


def log(line: str) -> None:
    print(line, flush=True)
    LOG.append(f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}  {line}")


def get(url: str, **kw) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.ok:
                return r
            log(f"HTTP {r.status_code} {url}")
        except requests.RequestException as e:  # noqa: PERF203
            log(f"error {url}: {e}")
        time.sleep(3 * (attempt + 1))
    return None


# ---------------------------------------------------------------- football-data.co.uk
def fetch_football_data(season: str) -> None:
    base = "https://www.football-data.co.uk"
    files = {f"{c}_{season}.csv": f"{base}/mmz4281/{season}/{c}.csv" for c in ("E0", "SP1", "I1", "D1", "F1")}
    files["fixtures.csv"] = f"{base}/fixtures.csv"
    for name, url in files.items():
        r = get(url)
        if r is None or len(r.content) < 200 or b"Div" not in r.content[:50]:
            log(f"KEPT    {name} (download failed or empty)")
            continue
        (OUT / name).write_bytes(r.content)
        log(f"ok      {name}  {len(r.content)} bytes")


# ---------------------------------------------------------------- ESPN (UEFA competitions)
COMPS = [("uefa.champions", "UCL"), ("uefa.europa", "UEL"), ("uefa.europa.conf", "UECL")]
FINAL = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}


def months(first: dt.date, last: dt.date) -> list[str]:
    out, y, m = [], first.year, first.month
    while (y, m) <= (last.year, last.month):
        if m not in (7, 8):  # qualifiers only
            out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def months_all(first: dt.date, last: dt.date) -> list[str]:
    out, y, m = [], first.year, first.month
    while (y, m) <= (last.year, last.month):
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# the five domestic leagues: results and goal timelines only (prices come from football-data.co.uk)
DOMESTIC = [("eng.1", "E0"), ("esp.1", "SP1"), ("ita.1", "I1"), ("ger.1", "D1"), ("fra.1", "F1")]


def minute(clock: dict) -> int | str:
    """Match minute of an event: from displayValue ("45'+2'" -> 47), else from the clock in seconds."""
    disp = str((clock or {}).get("displayValue") or "").replace("'", " ").replace("+", " ").split()
    try:
        return sum(int(float(x)) for x in disp[:2]) if disp else int(float((clock or {}).get("value", 0)) // 60) + 1
    except ValueError:
        return ""


def max_leads(details: list, home_id: str, hs: int, as_: int):
    """Largest lead of each side, the goal sequence ("12H;34A;81H", minute + scoring side) and the minute each side
    first led by two ("" if never). Own goals are tried both ways round: the version that reproduces the score wins."""
    goals = sorted(
        ({"clk": (d.get("clock") or {}).get("value", 0), "min": minute(d.get("clock") or {}), "team": str((d.get("team") or {}).get("id", "")),
          "og": bool(d.get("ownGoal"))}
         for d in details if d.get("scoringPlay")),
        key=lambda g: g["clk"])

    def run(flip: bool):
        h = a = mh = ma = 0
        seq, h2, a2 = [], "", ""
        for g in goals:
            for_home = g["team"] == home_id
            if g["og"] and flip:
                for_home = not for_home
            if for_home:
                h += 1
            else:
                a += 1
            seq.append(f"{g['min']}{'H' if for_home else 'A'}")
            if h - a >= 2 and h2 == "":
                h2 = g["min"]
            if a - h >= 2 and a2 == "":
                a2 = g["min"]
            mh, ma = max(mh, h - a), max(ma, a - h)
        return h, a, mh, ma, ";".join(seq), h2, a2

    for flip in (False, True):
        h, a, mh, ma, seq, h2, a2 = run(flip)
        if (h, a) == (hs, as_):
            return mh, ma, 1, seq, h2, a2
    return "", "", 0, "", "", ""


def fetch_espn(first: dt.date, last: dt.date, comps: list[tuple[str, str]] | None = None, skip_summer: bool = True) -> list[dict]:
    rows = []
    for lg, code in (comps or COMPS):
        for m in months(first, last) if skip_summer else months_all(first, last):
            r = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard", params={"dates": m, "limit": 300})
            if r is None:
                continue
            try:
                events = r.json().get("events", [])
            except ValueError:
                continue
            for e in events:
                c = e["competitions"][0]
                comp = {x.get("homeAway"): x for x in c.get("competitors", [])}
                h, a = comp.get("home"), comp.get("away")
                if not h or not a:
                    continue
                st = e["status"]["type"]["name"]
                hml = aml = ok = seq = h2 = a2 = ""
                hs, as_ = h.get("score"), a.get("score")
                if st in FINAL and hs not in (None, "") and as_ not in (None, ""):
                    hml, aml, ok, seq, h2, a2 = max_leads(c.get("details") or [], str(h["team"]["id"]), int(float(hs)), int(float(as_)))
                o = next((x for x in (c.get("odds") or []) if x and x.get("moneyline")), None)

                def ml(k):
                    v = (o or {}).get("moneyline", {}).get(k, {}) if o else {}
                    return (v.get("close") or {}).get("odds", "") if isinstance(v, dict) else ""
                rows.append({
                    "comp": code, "season": (e.get("season") or {}).get("year", ""), "event_id": e.get("id"), "date_utc": e.get("date"),
                    "status": st, "neutral": 1 if c.get("neutralSite") else 0, "home_id": h["team"]["id"], "home": h["team"]["displayName"],
                    "away_id": a["team"]["id"], "away": a["team"]["displayName"], "hs": hs, "as": as_, "home_max_lead": hml, "away_max_lead": aml,
                    "timeline_ok": ok, "goals": seq, "home_lead2_min": h2, "away_lead2_min": a2, "odds_provider": ((o or {}).get("provider") or {}).get("name", "") if o else "",
                    "ml_h": ml("home"), "ml_d": ml("draw"), "ml_a": ml("away"),
                })
            time.sleep(0.3)
        log(f"ok      ESPN {code}: {sum(1 for r in rows if r['comp'] == code)} events")
    return rows


def write_espn(rows: list[dict], name: str = "uefa_espn.csv", minimum: int = 500) -> None:
    cols = ["comp", "season", "event_id", "date_utc", "status", "neutral", "home_id", "home", "away_id", "away", "hs", "as",
            "home_max_lead", "away_max_lead", "timeline_ok", "goals", "home_lead2_min", "away_lead2_min", "odds_provider", "ml_h", "ml_d", "ml_a"]
    if len(rows) < minimum:  # a broken fetch must not wipe the file
        log(f"KEPT    {name} (only {len(rows)} events fetched)")
        return
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    log(f"ok      {name}  {len(rows)} events")


# ---------------------------------------------------------------- The Odds API (bet365 prices for Europe)
SPORTS = ["soccer_uefa_champs_league", "soccer_uefa_europa_league", "soccer_uefa_europa_conference_league"]


def fetch_odds(key: str, rows: list[dict], days_ahead: int = 4) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    soon = [r for r in rows if r["status"] == "STATUS_SCHEDULED" and r["date_utc"]
            and now <= dt.datetime.fromisoformat(r["date_utc"].replace("Z", "+00:00")) <= now + dt.timedelta(days=days_ahead)]
    if not soon:
        log("skip    odds (no European match in the next 4 days; saves the monthly quota)")
        return
    for sport in SPORTS:
        r = get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds/",
                params={"regions": "eu,uk", "markets": "h2h", "oddsFormat": "decimal", "apiKey": key})
        if r is None:
            log(f"KEPT    odds_{sport}.json (request failed)")
            continue
        try:
            events = r.json()
        except ValueError:
            log(f"KEPT    odds_{sport}.json (not JSON)")
            continue
        if not isinstance(events, list):
            log(f"KEPT    odds_{sport}.json ({str(events)[:80]})")
            continue
        (OUT / f"odds_{sport}.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
        log(f"ok      odds_{sport}.json  {len(events)} events, {r.headers.get('x-requests-remaining', '?')} requests left this month")


def main() -> None:
    today = dt.date.today()
    season = os.environ.get("FD_SEASON") or (f"{today.year % 100:02d}{(today.year + 1) % 100:02d}" if today.month >= 7 else f"{(today.year - 1) % 100:02d}{today.year % 100:02d}")
    fetch_football_data(season)
    rows = fetch_espn(dt.date(2024, 9, 1), today + dt.timedelta(days=75))
    write_espn(rows)
    # domestic goal timelines (who led by two, and when) for the early-payout check of the model: 2024-25 to date
    write_espn(fetch_espn(dt.date(2024, 8, 1), today, DOMESTIC, skip_summer=False), "domestic_espn.csv", minimum=1500)
    key = os.environ.get("ODDS_KEY", "").strip()
    if key:
        fetch_odds(key, rows)
    else:
        log("skip    odds (ODDS_KEY secret not set)")
    old = (OUT / "fetch.log").read_text(encoding="utf-8").splitlines() if (OUT / "fetch.log").exists() else []
    (OUT / "fetch.log").write_text("\n".join((old + LOG)[-400:]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
