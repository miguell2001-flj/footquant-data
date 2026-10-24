# footquant — runbook for the scheduled runs (read by Claude, not by Miguel)

Everything runs in the cloud. Nothing needs Miguel's PC.

- **This repository** holds the model (`footquant/`, `live_run.py`, `uefa_data.py`) and the data (`data/`), which a GitHub
  Actions workflow refreshes every morning and early afternoon (`fetch.py`): football-data.co.uk files for the five leagues,
  `uefa_espn.csv` (European results, goal timelines and fixtures) and, when the ODDS_KEY secret is set, `odds_*.json`
  (bet365 prices for Europe).
- **Footquant Desk** https://claude.ai/artifact/AsJiY8k7sHYWsq2Z3ZTc6X is the product and the record of bets. Its database
  (ArtifactData tool, `url` = that link) holds: collection `bets` (one document per suggestion, id = the `id` in
  `out/dash/bets.json`), collection `slips`, documents `meta/status`, `meta/prices` and `settings/main` (the rules).
- **Drive folder** `footquant` (Google Drive connector, folder id `1DTJnbFDnXxgq1gJk1chs7Q9fB3eApmKp`): a copy of the
  run outputs for Miguel to look at. Optional; never read from it.

Rules (from `settings/main`, set on the Desk): `edge_margin` is the required expected return per euro staked at the
price on offer (0.02 = 2%, the paid-out probability times the price minus one); `max_odds` the price cap (2.5 by
default since 22 Oct 2026); `kelly` quarter/half. The early-payout increment is calibrated to the realised
"led by two, did not win" rates (ValueConfig.early_payout_scale); `data/domestic_espn.csv` carries the goal
timelines, so domestic early payouts settle exactly. `underdog_check.py` + `underdog_report.py` reproduce the
22 Oct 2026 analysis (walk-forward pricing of the whole history by odds band).

**Pricing (since 23 Oct 2026)**: every domestic fixture is priced off the Betfair Exchange (football-data.co.uk
`BFE*` columns, margin removed by the power method), not off the model: the walk-forward fit gives the model a weight of
0.00-0.01 against the exchange. A bet365 home/away price qualifies when (exchange chance + early-payout chance) x price - 1
clears `edge_margin`; the early-payout chance (led by two, did not win; about 2%) is a curve fitted every run on the
goal timelines. `footquant/market/exchange.py` holds it. The run also writes a **price card** (`meta/prices`: every
fixture, its fair price and the lowest price worth taking at bet365 and elsewhere) and measures closing-line value against
the **exchange close**; `meta/status` carries the edge test (`clv_n`, `clv_mean`, `clv_lower95`, `clv_verdict`). When the
fixtures file holds no upcoming matches the run records nothing and says why in `data_warning` (no more dry runs).

**Slips (since 23 Oct 2026, setup "C2")**: besides the singles, each sheet run writes up to three documents to the
collection `slips`: a core double and treble built from the best value legs priced at or below `multi_max_leg` (2.00),
Kelly-sized on the combined price, and the fun-pot fold (`fun_legs` = 6 shortest bet365 favourites at 1.50 or shorter,
staked `fun_pot` = EUR 10 whatever the value). Slips only use legs starting at least an hour after the run. Every run
settles open slips (early payout counted on each leg). If the rules change before kick-off, an earlier run's open,
unplaced single that no longer passes is set to `status: "withdrawn"`.

**Comeback term (since 24 Oct 2026)**: the early-payout curve has a fourth term, the opponent's record of coming back
from two goals down ((came back + 1) / (trailed by two + 10), from earlier matches only, standardised and capped at
+-2). `meta/status.ep_curve_params` carries the fitted coefficients and every price-card side carries its `z`, so the
Desk can re-price with the same curve. Goal-minute study: odds ratio about 1.3 per standard deviation; a team's own
scoring timing showed no effect. `tests/test_comeback.py` checks it.

**Pre-kick-off check (since 24 Oct 2026)**: the workflow `prematch.yml` runs `prematch.py` every 15 minutes. For big-5
matches starting within 80 minutes it writes `data/prematch.json` (keyed by the model's `match_id`): the confirmed
line-ups from ESPN (changes since the last match, regulars out) and one Betfair Exchange + Pinnacle snapshot per league
taken 15-55 minutes before kick-off (The Odds API, about 1 credit a call, never below 40 credits left). The Desk's
**Kick-off check** reads that file through the viewer's Firecrawl connector and re-prices each open pick and slip leg.
Scheduled runs do not touch this; if `data/prematch.log` stops growing on match days, mention it in the report.

Tuesday run = bet sheet for the midweek round (Tue-Thu). Friday run = settle midweek, bet sheet for the weekend.
Monday run = settle the weekend. All use `--mode both`. European fixtures within 7 days are priced on every run.

## Steps

0. **Refresh the data first** (football-data.co.uk posts the weekend's fixtures on Friday, so the scheduled fetch may be
   stale): `gh api -X POST repos/<OWNER>/footquant-data/actions/workflows/fetch.yml/dispatches -f ref=main`, then poll
   `gh api "repos/<OWNER>/footquant-data/actions/runs?per_page=1" --jq '.workflow_runs[0].status'` every 20 s until it
   says `completed` (about 2 minutes). If `gh` is unavailable or the dispatch fails, say so in the report and go on.
1. **Clone**: `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/miguell2001-flj/footquant-data /home/claude/fq`
   (public; no credentials needed). Check `data/fetch.log` (last lines): if the newest `ok` lines are older
   than 36 hours, say so in the report and continue with the files as they are.
2. **Settings**: ArtifactData `get` `settings/main` and save its `data` object as `/home/claude/fq_docs/settings.json`
   (a python heredoc is fine). It holds Miguel's rules: `bankroll`, `edge_margin` (required edge, e.g. 0.02), `max_odds`
   (odds cap or null) and `kelly` ("quarter" or "half"). Missing fields mean the defaults (1000, 0.01, none, quarter).
3. **Export the record of bets**: ArtifactData `list` collection `bets` with `query.limit` 1000 and
   `out_dir` `/home/claude/fq_docs` (page with `query.cursor` while `next_cursor` is returned), and the same for the
   collection `slips` (it lands in `/home/claude/fq_docs/slips/`). The listing shows every
   document's `version`: write them to `/home/claude/fq_docs/versions.json` as `{"<doc_id>": <version>, ...,
   "meta/status": <version of meta/status from a get>, "meta/prices": <version of meta/prices from a get, if it exists>, "slips/<doc_id>": <version>, ...}`
   (a small python heredoc is fine).
4. **Run**: `pip install --break-system-packages numpy scipy pandas scikit-learn` if an import fails, then
   `python3 /home/claude/fq/live_run.py --data /home/claude/fq/data --out /home/claude/fq_out
   --settings /home/claude/fq_docs/settings.json --ledger-docs /home/claude/fq_docs --mode both`. The log says what happened (live sheet vs dry run, bets recorded,
   bets settled, European fixtures priced, stand-in vs bet365 prices).
5. **Update the Desk**: `cat /home/claude/fq_out/dash/db_batch_1.json` (and `_2`, `_3` ... if present) and pass each
   file's content as `writes` to ArtifactData `batch` (`url` = the Desk). The entries carry `if_version` already
   (from versions.json); a `null` one means the document is new and needs no version. On a version conflict re-read
   that document (`get`), put its version in and redo only that entry. Never write the fields `placed`,
   `placed_odds`, `placed_stake`, `placed_at` — Miguel sets those on the Desk, and the runner already leaves them out.
6. **Copy to Drive** (Google Drive connector; skip without comment if the connector is unavailable): for
   `sheet_<date>.csv`, `ledger.csv`, `ratings_<date>.csv`, `run_<date>.log` in `/home/claude/fq_out`:
   `search_files` with `parentId = '1DTJnbFDnXxgq1gJk1chs7Q9fB3eApmKp' and title = '<name>'`; `trash_file` the old one if
   found; `create_file` with `parentId` the folder id, `title` the name, `textContent` the file's content,
   `contentMimeType` `text/csv` (or `text/plain` for the log) and `disableConversionToGoogleType` true.
7. **Report** with SendUserMessage: the bet sheet as a table (date, league, match, bet on, odds, stake) — or
   "no bets this round" — plus, when European fixtures were priced from stand-in prices, one line saying how many are
   on the Desk's watch list waiting for a bet365 price; then one line on the ledger (bets settled, profit/loss, mean
   price edge vs the exchange close), one line per new slip (kind, legs, odds, real chance, stake) and one line on
   slips settled, one line with the edge test verdict (`clv_verdict` in the status), and the
   `data_warning` if there is one, ending with the Desk link. Nothing else.

If the clone fails or the Desk cannot be read, send one line saying what failed and stop.
