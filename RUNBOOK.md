# footquant — runbook for the scheduled runs (read by Claude, not by Miguel)

Everything runs in the cloud. Nothing needs Miguel's PC.

- **This repository** holds the model (`footquant/`, `live_run.py`, `uefa_data.py`) and the data (`data/`), which a GitHub
  Actions workflow refreshes every morning and early afternoon (`fetch.py`): football-data.co.uk files for the five leagues,
  `uefa_espn.csv` (European results, goal timelines and fixtures) and, when the ODDS_KEY secret is set, `odds_*.json`
  (bet365 prices for Europe).
- **Footquant Desk** https://claude.ai/artifact/AsJiY8k7sHYWsq2Z3ZTc6X is the product and the record of bets. Its database
  (ArtifactData tool, `url` = that link) holds: collection `bets` (one document per suggestion, id = the `id` in
  `out/dash/bets.json`), document `meta/status`, document `settings/main` (`bankroll`).
- **Drive folder** `footquant` (Google Drive connector, folder id `1DTJnbFDnXxgq1gJk1chs7Q9fB3eApmKp`): a copy of the
  run outputs for Miguel to look at. Optional; never read from it.

Tuesday run = bet sheet for the midweek round (Tue-Thu). Friday run = settle midweek, bet sheet for the weekend.
Monday run = settle the weekend. All use `--mode both`. European fixtures within 7 days are priced on every run.

## Steps

1. **Clone**: `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/miguell2001-flj/footquant-data /home/claude/fq`
   (public; no credentials needed). Check `data/fetch.log` (last lines): if the newest `ok` lines are older
   than 36 hours, say so in the report and continue with the files as they are.
2. **Bankroll**: ArtifactData `get` `settings/main` → `bankroll` (default 1000 if missing).
3. **Export the record of bets**: ArtifactData `list` collection `bets` with `query.limit` 1000 and
   `out_dir` `/home/claude/fq_docs` (page with `query.cursor` while `next_cursor` is returned). The listing shows every
   document's `version`: write them to `/home/claude/fq_docs/versions.json` as `{"<doc_id>": <version>, ...,
   "meta/status": <version of meta/status from a get>}` (a small python heredoc is fine).
4. **Run**: `pip install --break-system-packages numpy scipy pandas scikit-learn` if an import fails, then
   `python3 /home/claude/fq/live_run.py --data /home/claude/fq/data --out /home/claude/fq_out --bankroll <bankroll>
   --ledger-docs /home/claude/fq_docs --mode both`. The log says what happened (live sheet vs dry run, bets recorded,
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
   price edge), ending with the Desk link. Nothing else.

If the clone fails or the Desk cannot be read, send one line saying what failed and stop.
