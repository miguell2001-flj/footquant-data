"""Value detection against bookmaker prices (Phase 4).

For each selection s with quoted decimal odds o_s:

    p_raw_s   = 1 / o_s                       (raw implied)
    p_fair_s  = devig(o)_s                    (margin removed, market's "true" price)
    p_model_s = model probability of being *paid*
    edge_s    = p_model_s - p_fair_s
    EV_s      = p_model_s * (o_s - 1) - (1 - p_model_s) = p_model_s * o_s - 1   (per unit stake)

A bet is flagged when edge_s >= edge_margin (default 0.03). The stake module
then requires EV_s > 0 at the quoted price (Kelly is zero otherwise), so a
selection can be flagged yet receive no stake when the quoted margin eats the
whole edge - both numbers are reported.

Optional market shrinkage (ValueConfig.market_shrink = s): the probability used
for edge and staking becomes (1 - s) p_model + s p_fair. Against a sharp book
most apparent model edges are estimation noise; shrinking toward the market
removes the smallest ones and scales stakes down on the rest.

Early payout: for home/away selections in eligible competitions the
probability of being paid is P(win) + P(led by 2 and did not win); the second
term comes from `models.score_matrix.early_payout_probs`. Draw selections and
ineligible competitions use P(win) only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..config import ValueConfig
from .devig import devig, overround
from .odds_feed import OddsSnapshot

_MODEL_KEY = {
    ("1X2", "home"): "p_home",
    ("1X2", "draw"): "p_draw",
    ("1X2", "away"): "p_away",
    ("BTTS", "yes"): "p_btts_yes",
    ("BTTS", "no"): "p_btts_no",
    ("OU25", "over"): "p_over25",
    ("OU25", "under"): "p_under25",
}


@dataclass
class ValueBet:
    match_id: str
    bookmaker: str
    market: str
    selection: str
    odds: float
    p_raw: float
    p_fair: float
    p_model_win: float
    p_model_paid: float
    early_payout_bonus: float
    edge: float
    ev: float
    overround: float
    flagged: bool
    is_closing: bool

    def as_dict(self) -> dict:
        return asdict(self)


def expected_value(p: float, odds: float) -> float:
    """EV per unit stake of a binary bet paid at decimal `odds` with win prob p."""
    return p * (odds - 1.0) - (1.0 - p)


def evaluate_snapshot(
    probs: dict[str, float],
    snapshot: OddsSnapshot,
    cfg: ValueConfig | None = None,
    league: str | None = None,
    early_payout: dict[str, float] | None = None,
) -> list[ValueBet]:
    """Compare model probabilities with one bookmaker snapshot.

    `probs` is `DixonColesModel.predict` output (or the adjusted equivalent);
    `early_payout` is `early_payout_probs` output for the same fixture (ignored
    unless the market is 1X2 and the league is eligible).
    """
    cfg = cfg or ValueConfig()
    if snapshot.market not in cfg.markets:
        return []
    odds = snapshot.odds_vector
    fair = devig(odds, cfg.devig_method)
    ovr = overround(odds)
    eligible = (
        cfg.early_payout
        and snapshot.market == "1X2"
        and league is not None
        and league in cfg.early_payout_competitions
        and early_payout is not None
    )
    out: list[ValueBet] = []
    for sel, o, pf in zip(snapshot.selections, odds, fair):
        p_win = float(probs[_MODEL_KEY[(snapshot.market, sel)]])
        bonus = 0.0
        if eligible and sel in ("home", "away"):
            bonus = float(early_payout[f"p_{sel}_bonus"]) * cfg.early_payout_scale
        p_paid = min(p_win + bonus, 1.0)
        if cfg.market_shrink > 0:  # blend toward the market's own fair price
            p_paid = (1.0 - cfg.market_shrink) * p_paid + cfg.market_shrink * (float(pf) + bonus)
        edge = p_paid - float(pf)
        ev = expected_value(p_paid, o)
        out.append(
            ValueBet(
                match_id=snapshot.match_id,
                bookmaker=snapshot.bookmaker,
                market=snapshot.market,
                selection=sel,
                odds=float(o),
                p_raw=1.0 / float(o),
                p_fair=float(pf),
                p_model_win=p_win,
                p_model_paid=p_paid,
                early_payout_bonus=bonus,
                edge=edge,
                ev=ev,
                overround=ovr,
                flagged=edge >= cfg.edge_margin,
                is_closing=snapshot.is_closing,
            )
        )
    return out


def value_table(bets: list[ValueBet]) -> pd.DataFrame:
    if not bets:
        return pd.DataFrame(columns=[f.name for f in ValueBet.__dataclass_fields__.values()])
    return pd.DataFrame([b.as_dict() for b in bets])
