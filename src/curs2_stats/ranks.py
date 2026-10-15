"""Ranking with ties, as used by the non-parametric tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def average_ranks(x: ArrayLike) -> np.ndarray:
    """Rank values from 1 to n, giving tied values the mean of their positions.

    Parameters
    ----------
    x : array_like
        One-dimensional values to rank.

    Returns
    -------
    numpy.ndarray
        Ranks in the original order of ``x``.

    Examples
    --------
    The two 5s occupy positions 1 and 2, so both get (1 + 2) / 2 = 1.5:

    >>> average_ranks([5, 7, 5, 9]).tolist()
    [1.5, 3.0, 1.5, 4.0]

    Notes
    -----
    Averaging keeps the total of all ranks equal to ``n (n + 1) / 2``, which is
    the check done by hand after building the rank table.
    """
    values = np.asarray(x, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        # Positions are 1-based: start + 1 ... end + 1.
        ranks[order[start : end + 1]] = (start + 1 + end + 1) / 2
        start = end + 1
    return ranks
