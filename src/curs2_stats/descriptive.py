"""Descriptive statistics computed with the formulas used by hand."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Descriptives:
    """Summary of one sample.

    Attributes
    ----------
    n : int
        Sample size.
    mean : float
        Arithmetic mean, ``x_bar = sum(x_i) / n``.
    sum_of_squares : float
        ``SS = sum((x_i - x_bar)^2)``.
    variance : float
        Sample variance with Bessel's correction, ``s^2 = SS / (n - 1)``.
    std : float
        Sample standard deviation, ``s = sqrt(s^2)``.
    median : float
        Middle value of the sorted sample (mean of the two middle values
        when ``n`` is even).
    """

    n: int
    mean: float
    sum_of_squares: float
    variance: float
    std: float
    median: float

    @property
    def standard_error(self) -> float:
        """Standard error of the mean, ``SE = s / sqrt(n)``.

        It measures how precisely the mean is estimated, not how spread out
        the data are. It shrinks as ``n`` grows while ``s`` does not.
        """
        return self.std / sqrt(self.n)


def describe(x: ArrayLike) -> Descriptives:
    """Compute the descriptive statistics of a sample.

    Parameters
    ----------
    x : array_like
        One-dimensional sample with at least two values.

    Returns
    -------
    Descriptives

    Raises
    ------
    ValueError
        If the sample has fewer than two values (the variance needs n - 1 > 0).

    Notes
    -----
    The variance divides by ``n - 1`` rather than ``n``: the deviations are
    measured from the sample mean, which is itself fitted to the data, so they
    are slightly too small. Dividing by the degrees of freedom corrects this.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("describe() needs a one-dimensional sample with n >= 2.")

    n = int(values.size)
    mean = float(values.sum() / n)
    sum_of_squares = float(((values - mean) ** 2).sum())
    variance = sum_of_squares / (n - 1)
    return Descriptives(
        n=n,
        mean=mean,
        sum_of_squares=sum_of_squares,
        variance=variance,
        std=sqrt(variance),
        median=float(np.median(values)),
    )
