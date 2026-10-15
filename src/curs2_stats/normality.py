"""Normality tests computed by hand (course, Part 1).

Both tests have H0: "the data come from a normal distribution". Here a small
p-value is the "bad news": it means the data are NOT normal.

* Lilliefors: the Kolmogorov-Smirnov distance D, with the mean and the
  standard deviation estimated from the same sample and corrected critical
  values. Reject when D is large.
* Shapiro-Wilk: compares the sorted data with what a normal sample of the
  same size would look like. Reject when W is small (the reverse of D, t, z).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from curs2_stats.descriptive import describe


def standard_normal_cdf(z: ArrayLike) -> NDArray[np.float64]:
    """Phi(z), the cumulative distribution function of N(0, 1).

    Computed from the error function, ``Phi(z) = (1 + erf(z / sqrt 2)) / 2``,
    so it gives the same value as the normal table used on paper.
    """
    values = np.asarray(z, dtype=float)
    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(values)


@dataclass(frozen=True)
class LillieforsTable:
    """The table built by hand for the Lilliefors test.

    Each array has one entry per sorted observation ``x_(i)``.

    Attributes
    ----------
    x : numpy.ndarray
        Sorted sample.
    z : numpy.ndarray
        Standardised values, ``z_i = (x_(i) - x_bar) / s``.
    phi : numpy.ndarray
        Theoretical cumulative probability ``Phi(z_i)``.
    ecdf : numpy.ndarray
        Empirical cumulative proportion ``i / n``.
    d_plus : numpy.ndarray
        ``D+_i = i / n - Phi(z_i)`` (empirical curve above the normal one).
    d_minus : numpy.ndarray
        ``D-_i = Phi(z_i) - (i - 1) / n`` (empirical curve below the normal one).
    """

    x: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    ecdf: np.ndarray
    d_plus: np.ndarray
    d_minus: np.ndarray

    @property
    def d_plus_max(self) -> float:
        """``D+ = max(i/n - Phi(z_i))``: how far the step rises above the curve."""
        return float(self.d_plus.max())

    @property
    def d_minus_max(self) -> float:
        """``D- = max(Phi(z_i) - (i-1)/n)``: how far the step stays below the curve."""
        return float(self.d_minus.max())

    @property
    def statistic(self) -> float:
        """``D = max(D+, D-)``, always between 0 and 1."""
        return max(self.d_plus_max, self.d_minus_max)

    @property
    def argmax(self) -> float:
        """The observation where the largest distance D occurs."""
        idx = int(np.argmax(np.maximum(self.d_plus, self.d_minus)))
        return float(self.x[idx])

    def reject(self, d_critical: float) -> bool:
        """Reject normality when ``D > D_critical``."""
        return self.statistic > d_critical


def lilliefors_table(x: ArrayLike) -> LillieforsTable:
    """Build the Lilliefors table (x, z, Phi(z), i/n, D+, D-) for a sample.

    Parameters
    ----------
    x : array_like
        One-dimensional sample.

    Returns
    -------
    LillieforsTable

    Notes
    -----
    The Lilliefors test is the Kolmogorov-Smirnov test when the mean and the
    standard deviation of the normal distribution are estimated from the
    sample. ``D`` is the largest vertical gap between the empirical
    cumulative curve and the normal one. Both curves are probabilities, so
    ``0 <= D <= 1``; a value above 1 always signals a calculation error, such
    as comparing ``z`` itself instead of ``Phi(z)`` with ``i / n``.
    """
    stats = describe(x)
    sorted_x = np.sort(np.asarray(x, dtype=float))
    n = sorted_x.size
    i = np.arange(1, n + 1)

    z = (sorted_x - stats.mean) / stats.std
    phi = standard_normal_cdf(z)
    return LillieforsTable(
        x=sorted_x,
        z=z,
        phi=phi,
        ecdf=i / n,
        d_plus=i / n - phi,
        d_minus=phi - (i - 1) / n,
    )


@dataclass(frozen=True)
class ShapiroWilkSteps:
    """Intermediate values of the Shapiro-Wilk calculation.

    Attributes
    ----------
    x : numpy.ndarray
        Sorted sample.
    coefficients : numpy.ndarray
        Tabulated coefficients ``a_1 .. a_k`` with ``k = n // 2``.
    upper : numpy.ndarray
        ``x_(n+1-i)``, the values taken from the top of the sorted sample.
    lower : numpy.ndarray
        ``x_(i)``, the values taken from the bottom.
    differences : numpy.ndarray
        ``d_i = x_(n+1-i) - x_(i)``, pairs formed from the ends to the middle.
    weighted : numpy.ndarray
        ``a_i * d_i``.
    b : float
        ``b = sum(a_i * d_i)``.
    sum_of_squares : float
        ``Z^2 = sum((x_i - x_bar)^2)`` (the course's notation).
    """

    x: np.ndarray
    coefficients: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    differences: np.ndarray
    weighted: np.ndarray
    b: float
    sum_of_squares: float

    @property
    def statistic(self) -> float:
        """``W = b^2 / Z^2``, between 0 and 1; values close to 1 look normal."""
        return self.b**2 / self.sum_of_squares

    def reject(self, w_critical: float) -> bool:
        """Reject normality when ``W < W_critical``."""
        return self.statistic < w_critical


def shapiro_wilk_w(x: ArrayLike, coefficients: ArrayLike) -> ShapiroWilkSteps:
    """Compute the Shapiro-Wilk statistic W from tabulated coefficients.

    Parameters
    ----------
    x : array_like
        One-dimensional sample of size ``n``.
    coefficients : array_like
        Coefficients ``a_1 .. a_k`` for this ``n``, with ``k = n // 2``.

    Returns
    -------
    ShapiroWilkSteps

    Raises
    ------
    ValueError
        If the number of coefficients does not match ``n // 2``.

    Notes
    -----
    W compares two estimates of the spread: ``b``, built from the pairs of
    values weighted by the coefficients, and ``Z^2``. The largest weights go
    to the most extreme pairs, which is why the test is sensitive to long
    tails and asymmetry. Each n has its own row of coefficients; when n is
    odd, the middle value has no pair and does not enter ``b``.
    ``scipy.stats.shapiro`` approximates the coefficients with a formula, so
    its W can differ slightly from the table-based value.
    """
    sorted_x = np.sort(np.asarray(x, dtype=float))
    a = np.asarray(coefficients, dtype=float)
    k = sorted_x.size // 2
    if a.size != k:
        raise ValueError(
            f"Expected {k} coefficients for n = {sorted_x.size}, got {a.size}."
        )

    upper = sorted_x[::-1][:k]
    lower = sorted_x[:k]
    differences = upper - lower
    weighted = a * differences
    return ShapiroWilkSteps(
        x=sorted_x,
        coefficients=a,
        upper=upper,
        lower=lower,
        differences=differences,
        weighted=weighted,
        b=float(weighted.sum()),
        sum_of_squares=describe(sorted_x).sum_of_squares,
    )
