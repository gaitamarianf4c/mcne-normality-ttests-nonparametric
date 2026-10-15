"""Small exact helpers for p-values that can be computed without tables.

Kept private: they support the result objects of the public modules.
"""

from __future__ import annotations

from math import comb, erfc, sqrt


def normal_two_sided_p(z: float) -> float:
    """Two-sided p-value of a standard normal statistic, ``P(|Z| >= |z|)``."""
    return erfc(abs(z) / sqrt(2.0))


def binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided p-value of ``k`` successes in ``n`` trials with p = 0.5.

    Because Binomial(n, 0.5) is symmetric, the two-sided p-value is twice the
    probability of the smaller tail, capped at 1.
    """
    if n == 0:
        return 1.0
    smaller = min(k, n - k)
    tail = sum(comb(n, i) for i in range(smaller + 1)) / 2**n
    return min(1.0, 2.0 * tail)
