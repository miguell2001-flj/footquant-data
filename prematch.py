"""Pre-kick-off snapshot for the Desk's Kick-off check (run every 15 minutes by .github/workflows/prematch.yml).

For big-5 league fixtures (data/fixtures.csv) starting within the next 80 minutes:
  * line-ups: the confirmed starting XI from ESPN's match summary (once published, usually ~60 minutes before
    kick-off), the formation, how many starters changed since the team's last league match, and which regulars
    (started at least 2 of the last 3 league matches) are not starting;
  * prices: one Betfair Exchange + Pinnacle 1X2 snapshot per league from The Odds API, taken 15-55 minutes before
    kick-off (after the line-ups), one credit per league and kick-off slot; skipped when fewer than 40 credits remain.

Writes data/prematch.json (keyed by the model's match_id, kept for 2 days) and data/lineups_cache.json (starters of
finished matches, so each is fetched once). Usage: python prematch.py [--now 2026-10-24T13:20:00Z]
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import pathlib
import sys
from zoneinfo import ZoneInfo

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from uefa_data import _norm, match_name  # noqa: E402
from domestic_data import FD_TO_ESPN  # noqa: E402

DATA = HERE / "data"
OUT, CACHE, LOGF = DATA / "prematch.json", DATA / "lineups_cache.json", DATA / "prematch.log"
DIV_TO_LEAGUE = {"E0": "ENG-Premier League", "SP1": "ESP-La Liga", "I1": "ITA-Serie A", "D1": "GER-Bundesliga", "F1": "FRA-Ligue 1"}
SPORT = {"E0": "soccer_epl", "SP1": "soccer_spain_la_liga", "I1": "soccer_italy_serie_a", "D1": "soccer_germany_bundesliga", "F1": "soccer_france_ligue_one"}
ESPN = {"E0": "eng.1", "SP1": "esp.1", "I1": "ita.1", "D1": "ger.1", "F1": "fra.1"}
LOOKAHEAD_MIN, ODDS_FROM_MIN, ODDS_TO_MIN, MIN_CREDITS = 80, 55, 15, 40
UA = {"User-Agent": "Mozilla/5.0 (footquant prematch; github actions)"}
LOG: list[str] = []


def log(line: str, now: dt.datetime) -> None:
    print(line, flush=True)
    LOG.append(f"{now:%Y-%m-%dT%H:%M:%SZ}  {line}")


def get(url: str, **params):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=45)
        return r if r.ok else None
    except requests.RequestException:
        return None


def odds_get(sport: str, key: str, now: dt.datetime):
    """One h2h snapshot from The Odds API: Betfair Exchange (EU) and Pinnacle. Falls back to the whole EU region (which
    carries both) if the bookmaker filter is refused. Logs the status and the API's message on failure (never the key)."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    for extra in ({"bookmakers": "betfair_ex_eu,pinnacle"}, {"regions": "eu"}):
        try:
            r = requests.get(url, params={"apiKey": key, "markets": "h2h", "oddsFormat": "decimal", **extra}, headers=UA, timeout=45)
        except requests.RequestException as e:
            log(f"FAILED    prices {sport}: {type(e).__name__}", now)
            return None
        if r.ok:
            return r
        log(f"FAILED    prices {sport} ({'bookmakers' if 'bookmakers' in extra else 'regions'}): HTTP {r.status_code} {r.text[:160]!r}", now)
        if r.status_code not in (400, 422):
            return None
    return None


def season_label(d: dt.date) -> str:
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def fixtures(now: dt.datetime) -> list[dict]:
    """Big-5 fixtures from fixtures.csv with kick-off in UTC and the model's match_id (date and names as football-data)."""
    out = []
    with open(DATA / "fixtures.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            div = r.get("Div")
            if div not in DIV_TO_LEAGUE or not r.get("Date"):
                continue
            d = dt.datetime.strptime(r["Date"].strip(), "%d/%m/%Y")
            hh, mm = (r.get("Time") or "15:00").split(":")[:2]
            local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("Europe/London"))
            ko = local.astimezone(dt.timezone.utc)
            mid = f"{DIV_TO_LEAGUE[div]}|{season_label(d.date())}|{d.date()}|{r['HomeTeam']}|{r['AwayTeam']}"
            out.append({"div": div, "match_id": mid, "home": r["HomeTeam"], "away": r["AwayTeam"], "ko": ko})
    return out


def espn_name(fd: str, names: list[str]) -> str | None:
    cand = {_norm(n): n for n in names}
    fixed = FD_TO_ESPN.get(fd)
    return fixed if fixed in names else match_name(fd, cand)


def starters_of(summary: dict) -> dict:
    """{team_id: {"starters": [{"id", "name", "pos"}], "formation": str}} from an ESPN summary."""
    out = {}
    for t in summary.get("rosters") or []:
        st = [{"id": str(p["athlete"]["id"]), "name": p["athlete"].get("displayName", ""), "pos": (p.get("position") or {}).get("abbreviation", "")}
              for p in t.get("roster") or [] if p.get("starter")]
        out[str(t["team"]["id"])] = {"starters": st, "formation": t.get("formation") or ""}
    return out


def summary(div: str, event_id: str):
    r = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN[div]}/summary", event=event_id)
    try:
        return r.json() if r is not None else None
    except ValueError:
        return None


def past_events(team_id: str, before: dt.datetime, n: int = 3) -> list[tuple[str, str]]:
    """The team's last n finished league matches before `before`, from domestic_espn.csv: [(div, event_id)], newest first."""
    rows = []
    with open(DATA / "domestic_espn.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if team_id in (r["home_id"], r["away_id"]) and r["status"] != "STATUS_SCHEDULED" and r["hs"] != "":
                t = dt.datetime.fromisoformat(r["date_utc"].replace("Z", "+00:00"))
                if t < before:
                    rows.append((t, r["comp"], r["event_id"]))
    rows.sort(reverse=True)
    return [(c, e) for _, c, e in rows[:n]]


def lineup_report(team_id: str, today: dict, cache: dict, ko: dt.datetime) -> dict:
    """Today's XI against the team's last three league XIs: starters changed since the last match, and regulars
    (started at least two of the last three) who are not starting. Finished matches are cached by event id."""
    prev, names = [], {}
    for c, e in past_events(team_id, ko):
        if e not in cache:
            s = summary(c, e)
            if s is None:
                continue
            cache[e] = {tid: [[p["id"], p["name"]] for p in v["starters"]] for tid, v in starters_of(s).items()}
        xi = cache[e].get(team_id, [])
        names.update({pid: nm for pid, nm in xi})
        prev.append([pid for pid, _ in xi])
    ids = {p["id"] for p in today["starters"]}
    count = {}
    for xi in prev:
        for pid in xi:
            count[pid] = count.get(pid, 0) + 1
    regular = {pid for pid, k in count.items() if k >= 2}
    return {"formation": today["formation"], "starters": [p["name"] for p in today["starters"]],
            "changes": (11 - len(ids & set(prev[0]))) if prev and prev[0] else None,
            "regulars_out": sorted(names.get(pid, pid) for pid in regular - ids),
            "matches_compared": len(prev)}


def main() -> int:
    LOG.clear()
    args = sys.argv[1:]
    now = dt.datetime.fromisoformat(args[args.index("--now") + 1].replace("Z", "+00:00")) if "--now" in args else dt.datetime.now(dt.timezone.utc)
    state = json.loads(OUT.read_text()) if OUT.exists() else {"matches": {}}
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    # keep two days of history
    state["matches"] = {k: v for k, v in state.get("matches", {}).items()
                        if dt.datetime.fromisoformat(v["kickoff_utc"].replace("Z", "+00:00")) > now - dt.timedelta(days=2)}
    fx = fixtures(now)
    due = [f for f in fx if now < f["ko"] <= now + dt.timedelta(minutes=LOOKAHEAD_MIN)]
    if not due:
        print("nothing due")
        return 0
    changed = False
    # ---- line-ups (ESPN)
    boards = {}
    for f in due:
        key = (f["div"], f["ko"].strftime("%Y%m%d"))
        if key not in boards:
            r = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN[f['div']]}/scoreboard", dates=key[1])
            try:
                boards[key] = r.json().get("events", []) if r is not None else []
            except ValueError:
                boards[key] = []
        ev = None
        for e in boards[key]:
            comp = {x.get("homeAway"): x for x in e["competitions"][0].get("competitors", [])}
            if "home" in comp and "away" in comp:
                hn, an = comp["home"]["team"]["displayName"], comp["away"]["team"]["displayName"]
                if espn_name(f["home"], [hn, an]) == hn and espn_name(f["away"], [hn, an]) == an:
                    ev = (e["id"], str(comp["home"]["team"]["id"]), str(comp["away"]["team"]["id"]))
                    break
        entry = state["matches"].setdefault(f["match_id"], {"div": f["div"], "home": f["home"], "away": f["away"],
                                                            "kickoff_utc": f["ko"].strftime("%Y-%m-%dT%H:%M:%SZ")})
        if ev is None:
            entry["lineups"] = {"confirmed": False, "note": "match not found on ESPN"}
            continue
        if (entry.get("lineups") or {}).get("confirmed"):
            continue
        s = summary(f["div"], ev[0])
        xi = starters_of(s) if s else {}
        home, away = xi.get(ev[1]), xi.get(ev[2])
        if not home or not away or len(home["starters"]) < 11 or len(away["starters"]) < 11:
            entry["lineups"] = {"confirmed": False, "note": "line-ups not published yet"}
            changed = True
            continue
        rep = {side: lineup_report(tid, today, cache, f["ko"]) for side, tid, today in (("home", ev[1], home), ("away", ev[2], away))}
        entry["lineups"] = {"confirmed": True, "taken_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), **rep}
        changed = True
        log(f"line-ups  {f['home']} - {f['away']}: changes {rep['home']['changes']}/{rep['away']['changes']}, regulars out "
            f"{', '.join(rep['home']['regulars_out']) or '-'} / {', '.join(rep['away']['regulars_out']) or '-'}", now)
    # ---- prices (The Odds API), one call per league with a match 15-55 minutes away and no snapshot yet in that window
    key = os.environ.get("ODDS_KEY", "").strip()
    leagues = sorted({f["div"] for f in due
                      if dt.timedelta(minutes=ODDS_TO_MIN) <= f["ko"] - now <= dt.timedelta(minutes=ODDS_FROM_MIN)
                      and not ((state["matches"].get(f["match_id"]) or {}).get("exchange") or {}).get("taken_utc", "") >= (f["ko"] - dt.timedelta(minutes=ODDS_FROM_MIN + 5)).strftime("%Y-%m-%dT%H:%M:%SZ")})
    credits = state.get("credits_left")
    last_call = state.setdefault("last_call", {})
    for div in leagues:
        if last_call.get(div, "") > (now - dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"):
            continue  # at most one call per league every 30 minutes, even if a fixture could not be matched
        if not key:
            log("skip      prices (ODDS_KEY not set)", now)
            break
        if credits is not None and int(credits) < MIN_CREDITS:
            log(f"skip      prices ({credits} credits left, keeping {MIN_CREDITS} in reserve)", now)
            break
        r = odds_get(SPORT[div], key, now)
        if r is None:
            changed = True  # keep the failure in the log
            continue
        last_call[div] = now.strftime("%Y-%m-%dT%H:%M:%SZ")  # failed calls cost no credits, so only a success starts the 30-minute wait
        credits = r.headers.get("x-requests-remaining", credits)
        cands = {_norm(n): n for f in fx if f["div"] == div for n in (f["home"], f["away"])}
        n_ok = 0
        for e in r.json():
            h, a = match_name(e["home_team"], cands), match_name(e["away_team"], cands)
            t = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            f = next((f for f in fx if f["div"] == div and f["home"] == h and f["away"] == a and abs((f["ko"] - t).total_seconds()) < 3 * 3600), None)
            if f is None or not (now < f["ko"] <= now + dt.timedelta(hours=6)):
                continue
            entry = state["matches"].setdefault(f["match_id"], {"div": div, "home": f["home"], "away": f["away"], "kickoff_utc": f["ko"].strftime("%Y-%m-%dT%H:%M:%SZ")})
            for b in e.get("bookmakers", []):
                if b.get("key") not in ("betfair_ex_eu", "pinnacle"):
                    continue  # the regions fallback returns every EU bookmaker
                mk = next((m for m in b.get("markets", []) if m["key"] == "h2h"), None)
                if not mk:
                    continue
                pr = {o["name"]: float(o["price"]) for o in mk["outcomes"]}
                trio = {"home": pr.get(e["home_team"]), "draw": pr.get("Draw"), "away": pr.get(e["away_team"])}
                if all(trio.values()):
                    entry["exchange" if b["key"] == "betfair_ex_eu" else "pinnacle"] = {**trio, "taken_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
                    n_ok += b["key"] == "betfair_ex_eu"
        changed = True
        log(f"prices    {div}: exchange prices for {n_ok} matches; {credits} credits left", now)
    state["updated_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    state["credits_left"] = credits
    if changed:
        OUT.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        old = LOGF.read_text(encoding="utf-8").splitlines() if LOGF.exists() else []
        LOGF.write_text("\n".join((old + LOG)[-300:]) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
