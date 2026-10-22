"""Removing the bookmaker margin (overround) from a set of decimal odds.

Raw implied probabilities pi_i = 1 / o_i sum to beta = 1 + margin. Three ways
to recover a fair vector p with sum(p) = 1:

multiplicative  p_i = pi_i / beta
    Assumes the margin is spread in proportion to probability. Simple, but
    systematically over-states longshot probabilities because bookmakers load
    extra margin onto longshots (the favourite-longshot bias).

power           p_i = pi_i ** k,  k > 1 solving sum_i pi_i ** k = 1
    Clarke, Kovalchik & Ingram (2017, Am. J. Sports Sci. 5(6)) - margin is
    removed disproportionately from longshots.

shin            p_i = [ sqrt(z^2 + 4 (1 - z) pi_i^2 / beta) - z ] / (2 (1 - z))
    with z in (0, 1) solving sum_i p_i = 1. From Shin (1993, Econ. J. 103) via
    Jullien & Salanie (1994): z is the share of insider money the bookmaker
    prices against; it produces the same longshot correction as the power
    method from a market-microstructure argument. Strumbelj (2014, Int. J.
    Forecasting 30(4)) finds Shin the most accurate of the standard methods
    for football 1X2 markets, which is why it is the default.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied_probabilities(odds) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("decimal odds must be > 1")
    return 1.0 / odds


def overround(odds) -> float:
    """Bookmaker margin: sum of raw implied probabilities minus one."""
    return float(implied_probabilities(odds).sum() - 1.0)


def _shin(pi: np.ndarray) -> np.ndarray:
    beta = pi.sum()

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / beta) - z) / (2.0 * (1.0 - z))

    f = lambda z: probs(z).sum() - 1.0
    if f(0.0) <= 0.0:  # no margin (or negative) - nothing to remove
        return pi / beta
    z = brentq(f, 0.0, 0.999, xtol=1e-12)
    p = probs(z)
    return p / p.sum()


def _power(pi: np.ndarray) -> np.ndarray:
    f = lambda k: np.sum(pi**k) - 1.0
    if f(1.0) <= 0.0:
        return pi / pi.sum()
    k = brentq(f, 1.0, 50.0, xtol=1e-12)
    p = pi**k
    return p / p.sum()


def devig(odds, method: str = "shin") -> np.ndarray:
    """Fair probabilities for one market's full set of mutually exclusive prices."""
    pi = implied_probabilities(odds)
    if method == "multiplicative":
        return pi / pi.sum()
    if method == "power":
        return _power(pi)
    if method == "shin":
        return _shin(pi)
    raise ValueError("method must be 'shin', 'power' or 'multiplicative'")
