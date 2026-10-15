"""One group compared with a reference value (course, Part 2).

The same functions handle paired data (course, Part 3): the paired test in
:mod:`curs2_stats.two_sample` turns the two measurements into one column of
differences and tests it against 0.

Every statistic follows the course idea *signal divided by noise*.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike

from curs2_stats._distributions import binomial_two_sided_p, normal_two_sided_p
from curs2_stats.descriptive import describe
from curs2_stats.ranks import average_ranks


@dataclass(frozen=True)
class TTestResult:
    """Result of a t-test.

    Attributes
    ----------
    statistic : float
        ``t = estimate / standard_error`` (signal / noise).
    df : int
        Degrees of freedom.
    estimate : float
        The signal: mean minus reference, or difference of means.
    standard_error : float
        The noise: how much ``estimate`` would vary by chance alone.
    cohen_d : float
        Effect size ``|estimate| / s``, independent of the sample size.
        Rough guide: 0.2 small, 0.5 medium, 0.8 or more large.
    """

    statistic: float
    df: int
    estimate: float
    standard_error: float
    cohen_d: float

    def reject(self, t_critical: float) -> bool:
        """Two-sided decision: reject H0 when ``|t| > t_critical``."""
        return abs(self.statistic) > t_critical

    def confidence_interval(self, t_critical: float) -> tuple[float, float]:
        """Confidence interval of ``estimate``: ``estimate +/- t_critical * SE``.

        With the 95 % critical value, the interval excludes 0 exactly when the
        two-sided test at alpha = 0.05 rejects H0.
        """
        margin = t_critical * self.standard_error
        return (self.estimate - margin, self.estimate + margin)


def one_sample_t(x: ArrayLike, mu0: float) -> TTestResult:
    """One-sample t-test of H0: mu = mu0.

    ``t = (x_bar - mu0) / (s / sqrt(n))`` with ``df = n - 1``: how many
    standard errors separate the observed mean from the reference value.

    Assumption: the data are approximately normal (check with Shapiro-Wilk
    and a histogram first).

    Parameters
    ----------
    x : array_like
        Sample.
    mu0 : float
        Reference value under H0.
    """
    stats = describe(x)
    estimate = stats.mean - mu0
    return TTestResult(
        statistic=estimate / stats.standard_error,
        df=stats.n - 1,
        estimate=estimate,
        standard_error=stats.standard_error,
        cohen_d=abs(estimate) / stats.std,
    )


@dataclass(frozen=True)
class SignTestResult:
    """Result of the sign test.

    Attributes
    ----------
    n_plus, n_minus, n_ties : int
        Values above, below and equal to the reference. Ties are discarded.
    p_value : float
        Exact two-sided p-value from Binomial(n_plus + n_minus, 0.5).
    """

    n_plus: int
    n_minus: int
    n_ties: int
    p_value: float

    @property
    def n(self) -> int:
        """Number of non-tied observations used by the test."""
        return self.n_plus + self.n_minus

    @property
    def z(self) -> float:
        """Normal approximation ``z = (n_plus - n/2) / sqrt(n/4)``.

        Good when ``n p >= 5``; for small samples the exact binomial
        p-value (:attr:`p_value`) is the one to use.
        """
        return (self.n_plus - self.n / 2) / sqrt(self.n / 4)

    def reject(self, alpha: float) -> bool:
        """Reject H0 when ``p < alpha``."""
        return self.p_value < alpha


def sign_test(x: ArrayLike, mu0: float) -> SignTestResult:
    """Sign test of H0: median = mu0.

    Each value gets a "+" if it is above ``mu0`` and a "-" if it is below.
    If ``mu0`` were really the middle of the distribution, every value would
    fall above or below it with probability 0.5, like a fair coin, so the
    number of "+" signs follows Binomial(n, 0.5). Nothing is assumed about
    the shape of the distribution.
    """
    differences = np.asarray(x, dtype=float) - mu0
    n_plus = int((differences > 0).sum())
    n_minus = int((differences < 0).sum())
    return SignTestResult(
        n_plus=n_plus,
        n_minus=n_minus,
        n_ties=int((differences == 0).sum()),
        p_value=binomial_two_sided_p(n_plus, n_plus + n_minus),
    )


@dataclass(frozen=True)
class WilcoxonResult:
    """Result of the Wilcoxon signed-rank test.

    Attributes
    ----------
    differences : numpy.ndarray
        Non-zero differences ``d_i = x_i - mu0``.
    ranks : numpy.ndarray
        Average ranks of ``|d_i|``.
    w_plus, w_minus : float
        Sums of the ranks of the positive and of the negative differences.
    """

    differences: np.ndarray
    ranks: np.ndarray
    w_plus: float
    w_minus: float

    @property
    def n(self) -> int:
        """Number of non-zero differences."""
        return int(self.differences.size)

    @property
    def statistic(self) -> float:
        """``W = min(W+, W-)``, compared with the Wilcoxon table."""
        return min(self.w_plus, self.w_minus)

    @property
    def z(self) -> float:
        """Normal approximation from the course, ``z = (W+ - mu) / sigma``.

        ``mu = n (n + 1) / 4`` and ``sigma = sqrt(n (n + 1)(2n + 1) / 24)``.
        Recommended for n above about 20; for smaller samples use the table.
        Software additionally corrects sigma for ties.
        """
        mu = self.n * (self.n + 1) / 4
        sigma = sqrt(self.n * (self.n + 1) * (2 * self.n + 1) / 24)
        return (self.w_plus - mu) / sigma

    @property
    def p_value_normal(self) -> float:
        """Two-sided p-value of the normal approximation :attr:`z`."""
        return normal_two_sided_p(self.z)

    def reject(self, w_critical: float) -> bool:
        """Reject H0 when ``W <= W_critical`` (a small W means unbalanced signs)."""
        return self.statistic <= w_critical


def wilcoxon_signed_rank(x: ArrayLike, mu0: float = 0.0) -> WilcoxonResult:
    """Wilcoxon signed-rank test of H0: the differences are centred on 0.

    Steps, as in the course:

    1. ``d_i = x_i - mu0``; zero differences are removed.
    2. ``|d_i|`` is ranked from 1 (smallest) upwards; equal values get the
       mean of the ranks they occupy.
    3. ``W+`` = sum of the ranks with ``d_i > 0``, ``W-`` = sum with ``d_i < 0``;
       check ``W+ + W- = n (n + 1) / 2``.
    4. Under H0 the two sums are close; one sum much smaller than the other
       leads to rejecting H0.

    Unlike the sign test, the size of each difference matters, not only its
    direction. Assumption: the distribution is roughly symmetric, which is
    weaker than normality.
    """
    differences = np.asarray(x, dtype=float) - mu0
    differences = differences[differences != 0]
    ranks = average_ranks(np.abs(differences))
    return WilcoxonResult(
        differences=differences,
        ranks=ranks,
        w_plus=float(ranks[differences > 0].sum()),
        w_minus=float(ranks[differences < 0].sum()),
    )
