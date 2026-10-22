# footquant-data

Data and model for Miguel's football paper-betting desk. The GitHub Actions workflow in `.github/workflows/fetch.yml`
refreshes `data/` twice a day (football-data.co.uk results and prices for England, Spain, Italy, Germany and France;
Champions, Europa and Conference League results and fixtures from ESPN; bet365 prices for Europe from The Odds API
when the `ODDS_KEY` repository secret is set). Claude's scheduled runs clone this repository, price the coming fixtures
and update the Footquant Desk. `RUNBOOK.md` describes those runs.

Nothing here is private: public match data, public prices, and the model code.
