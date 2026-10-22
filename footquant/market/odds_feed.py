"""Odds ingestion (Phase 4).

The value engine consumes `OddsSnapshot` objects - one bookmaker, one market,
one match, the full set of prices for that market at one timestamp. Where the
prices come from is behind the `OddsFeed` protocol:

* `snapshots_from_frame`  - any DataFrame in the ODDS_COLUMNS layout (this is
  what `data.ingest.load_football_data_csv` returns for historical bet365
  opening and closing lines, and what a licensed live feed should be mapped to).
* `JsonOddsFeed`          - a saved JSON export in the same layout, for feeds
  you poll yourself.
* `HttpJsonOddsFeed`      - a thin GET wrapper (API key from an environment
  variable) around any odds API that returns the same record layout, with a
  `mapper` callable to translate the provider's schema.

bet365 is not among The Odds API's UK/EU bookmakers (only `bet365_au`, AFL/NRL),
so live bet365 lines need a provider that licenses them (e.g. BetsAPI,
odds-api.io, Sportmonks) or a manual snapshot before kick-off. Automated
collection from bet365.com itself is against its terms of use and actively
blocked; this module deliberately does not include such a scraper.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

import pandas as pd

from ..data.schema import MARKET_SELECTIONS


@dataclass
class OddsSnapshot:
    match_id: str
    bookmaker: str
    market: str  # "1X2" | "BTTS" | "OU25"
    prices: dict[str, float]  # selection -> decimal odds
    timestamp: pd.Timestamp | None = None
    is_closing: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.market not in MARKET_SELECTIONS:
            raise ValueError(f"unknown market {self.market!r}")
        expected = set(MARKET_SELECTIONS[self.market])
        if set(self.prices) != expected:
            raise ValueError(f"{self.market} snapshot needs prices for {sorted(expected)}, got {sorted(self.prices)}")
        for sel, o in self.prices.items():
            if not o or o <= 1.0:
                raise ValueError(f"invalid odds {o} for {sel}")

    @property
    def selections(self) -> tuple[str, ...]:
        return MARKET_SELECTIONS[self.market]

    @property
    def odds_vector(self) -> list[float]:
        return [self.prices[s] for s in self.selections]


class OddsFeed(Protocol):
    def fetch(self) -> list[OddsSnapshot]: ...


def snapshots_from_frame(odds: pd.DataFrame, bookmaker: str | None = "bet365") -> list[OddsSnapshot]:
    """Group an ODDS_COLUMNS frame into complete-market snapshots.

    Incomplete markets (e.g. a 1X2 with only two prices) are skipped.
    """
    df = odds.copy()
    if bookmaker is not None:
        df = df[df["bookmaker"] == bookmaker]
    if "is_closing" not in df.columns:
        df["is_closing"] = False
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.NaT
    out: list[OddsSnapshot] = []
    keys = ["match_id", "bookmaker", "market", "is_closing", "timestamp"]
    for (mid, bk, mkt, closing, ts), grp in df.groupby(keys, dropna=False, sort=False):
        prices = dict(zip(grp["selection"], grp["odds"].astype(float)))
        if set(prices) != set(MARKET_SELECTIONS.get(mkt, ())):
            continue
        out.append(
            OddsSnapshot(
                match_id=str(mid),
                bookmaker=str(bk),
                market=str(mkt),
                prices=prices,
                timestamp=None if pd.isna(ts) else pd.Timestamp(ts),
                is_closing=bool(closing),
            )
        )
    return out


class JsonOddsFeed:
    """Records: [{match_id, bookmaker, market, selection, odds, timestamp, is_closing}, ...]."""

    def __init__(self, path: str | Path, bookmaker: str = "bet365") -> None:
        self.path = Path(path)
        self.bookmaker = bookmaker

    def fetch(self) -> list[OddsSnapshot]:
        with open(self.path, encoding="utf-8") as fh:
            records = json.load(fh)
        return snapshots_from_frame(pd.DataFrame(records), self.bookmaker)


class HttpJsonOddsFeed:
    """GET an odds API and map its payload into snapshots.

    `mapper(payload) -> Iterable[dict]` must yield ODDS_COLUMNS records; the
    provider's own schema (event ids, team names, market keys) is the mapper's
    job, together with the team-alias map used for the match table.
    """

    def __init__(
        self,
        url: str,
        mapper: Callable[[object], Iterable[dict]],
        api_key_env: str | None = None,
        params: dict | None = None,
        bookmaker: str = "bet365",
        timeout: float = 20.0,
    ) -> None:
        self.url = url
        self.mapper = mapper
        self.api_key_env = api_key_env
        self.params = dict(params or {})
        self.bookmaker = bookmaker
        self.timeout = timeout

    def fetch(self) -> list[OddsSnapshot]:
        import requests  # local import keeps the dependency optional

        params = dict(self.params)
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"set {self.api_key_env} in the environment")
            params["apiKey"] = key
        resp = requests.get(self.url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        records = list(self.mapper(resp.json()))
        return snapshots_from_frame(pd.DataFrame(records), self.bookmaker)
