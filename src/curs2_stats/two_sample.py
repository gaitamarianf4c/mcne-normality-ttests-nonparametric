"""Two sets of measurements (course, Parts 3 and 4).

The first question is always the design. The course's control question: does
it make sense to draw a line between a particular value of the first group and
a particular value of the second?

* **yes - paired data**: the same patients, datasets or images measured twice;
  analyse the column of differences;
* **no - independent groups**: different individuals; compare the groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike

from curs2_stats._distributions import binomial_two_sided_p, normal_two_sided_p
from curs2_stats.descriptive import describe
from curs2_stats.one_sample import TTestResult, one_sample_t
from curs2_stats.ranks import average_ranks


def paired_t(after: ArrayLike, before: ArrayLike) -> TTestResult:
    """Paired t-test of H0: the mean of the differences ``after - before`` is 0.

    The two columns become one column of differences, tested with the
    one-sample t-test against 0: ``t = d_bar / (s_d / sqrt(n))``, ``df = n - 1``.
    Each unit is its own control, so the differences *between* units no
    longer add noise.

    Normality is checked on the differences, not on the two columns.

    Raises
    ------
    ValueError
        If the two samples do not have the same length.
    """
    after_values = np.asarray(after, dtype=float)
    before_values = np.asarray(before, dtype=float)
    if after_values.shape != before_values.shape:
        raise ValueError("Paired samples must have the same length.")
    return one_sample_t(after_values - before_values, mu0=0.0)


@dataclass(frozen=True)
class IndependentTResult(TTestResult):
    """Result of Student's t-test for two independent groups.

    Attributes
    ----------
    pooled_std : float
        ``s_p``, the common standard deviation of the two groups.
    std_ratio : float
        Larger s / smaller s. The course's rule of thumb: below 2, the
        variances can be treated as close.
    variance_ratio : float
        Larger s^2 / smaller s^2 (a stricter version of the same check).
    """

    pooled_std: float = 0.0
    std_ratio: float = 1.0
    variance_ratio: float = 1.0


def independent_t(a: ArrayLike, b: ArrayLike) -> IndependentTResult:
    """Student's t-test for two independent groups with pooled variance.

    Formulas, as in the course:

    * ``s_p^2 = ((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2)``
    * ``SE = s_p * sqrt(1 / n_a + 1 / n_b)``
    * ``t = (x_bar_a - x_bar_b) / SE``, ``df = n_a + n_b - 2``

    Conditions: independent observations, normality in each group, close
    variances. When the variances differ, use Welch's version
    (``scipy.stats.ttest_ind(..., equal_var=False)``).
    """
    stats_a = describe(a)
    stats_b = describe(b)
    df = stats_a.n + stats_b.n - 2
    pooled_variance = (
        (stats_a.n - 1) * stats_a.variance + (stats_b.n - 1) * stats_b.variance
    ) / df
    pooled_std = sqrt(pooled_variance)
    standard_error = pooled_std * sqrt(1 / stats_a.n + 1 / stats_b.n)
    estimate = stats_a.mean - stats_b.mean
    small_s, large_s = sorted((stats_a.std, stats_b.std))

    return IndependentTResult(
        statistic=estimate / standard_error,
        df=df,
        estimate=estimate,
        standard_error=standard_error,
        cohen_d=abs(estimate) / pooled_std,
        pooled_std=pooled_std,
        std_ratio=large_s / small_s,
        variance_ratio=(large_s / small_s) ** 2,
    )


@dataclass(frozen=True)
class MannWhitneyResult:
    """Result of the Mann-Whitney test.

    Attributes
    ----------
    ranks_a, ranks_b : numpy.ndarray
        Ranks of each group within the combined, sorted sample.
    rank_sum_a, rank_sum_b : float
        ``R_a`` and ``R_b``; check ``R_a + R_b = N (N + 1) / 2``.
    u_a, u_b : float
        ``U = R - n (n + 1) / 2``: how many times a value of this group
        exceeds a value of the other group (ties count one half).
        Check ``U_a + U_b = n_a n_b``.
    z : float
        Normal approximation from the course, based on the rank sum ``T`` of
        the smaller group.
    """

    ranks_a: np.ndarray
    ranks_b: np.ndarray
    rank_sum_a: float
    rank_sum_b: float
    u_a: float
    u_b: float
    z: float

    @property
    def statistic(self) -> float:
        """``U = min(U_a, U_b)``, compared with the Mann-Whitney table."""
        return min(self.u_a, self.u_b)

    @property
    def p_value_normal(self) -> float:
        """Two-sided p-value of the normal approximation :attr:`z`."""
        return normal_two_sided_p(self.z)

    def reject(self, u_critical: float) -> bool:
        """Reject H0 when ``U <= U_critical`` (a small U means separated groups)."""
        return self.statistic <= u_critical


def mann_whitney(a: ArrayLike, b: ArrayLike) -> MannWhitneyResult:
    """Mann-Whitney test for two independent groups.

    Steps, as in the course:

    1. Put the observations of both groups together.
    2. Sort them and rank them; equal values get the mean rank.
    3. Add the ranks of each group: ``R_a``, ``R_b``.
    4. If a group collects much larger or much smaller ranks than its share,
       reject H0.

    No normality assumption is needed: the test works with ranks, so one
    extreme value cannot fool it.

    Normal approximation (course): ``T`` is the rank sum of the smaller group
    (size ``n_s``), ``mu = n_s (N + 1) / 2``,
    ``sigma = sqrt(n_a n_b (N + 1) / 12)`` and ``z = (T - mu) / sigma``.
    Software makes sigma slightly smaller when there are ties.
    """
    a_values = np.asarray(a, dtype=float)
    b_values = np.asarray(b, dtype=float)
    n_a, n_b = a_values.size, b_values.size
    n_total = n_a + n_b

    ranks = average_ranks(np.concatenate([a_values, b_values]))
    ranks_a, ranks_b = ranks[:n_a], ranks[n_a:]
    rank_sum_a = float(ranks_a.sum())
    rank_sum_b = float(ranks_b.sum())

    if n_a <= n_b:
        t_small, n_small = rank_sum_a, n_a
    else:
        t_small, n_small = rank_sum_b, n_b
    mu = n_small * (n_total + 1) / 2
    sigma = sqrt(n_a * n_b * (n_total + 1) / 12)

    return MannWhitneyResult(
        ranks_a=ranks_a,
        ranks_b=ranks_b,
        rank_sum_a=rank_sum_a,
        rank_sum_b=rank_sum_b,
        u_a=rank_sum_a - n_a * (n_a + 1) / 2,
        u_b=rank_sum_b - n_b * (n_b + 1) / 2,
        z=(t_small - mu) / sigma,
    )


@dataclass(frozen=True)
class McNemarResult:
    """Result of the exact McNemar test.

    Attributes
    ----------
    only_first_correct : int
        Units where only the first model is correct.
    only_second_correct : int
        Units where only the second model is correct.
    p_value : float
        Exact two-sided p-value from Binomial(discordant units, 0.5).
    """

    only_first_correct: int
    only_second_correct: int
    p_value: float

    def reject(self, alpha: float) -> bool:
        """Reject H0 when ``p < alpha``."""
        return self.p_value < alpha


def mcnemar_exact(first_correct: ArrayLike, second_correct: ArrayLike) -> McNemarResult:
    """Exact McNemar test for two models evaluated on the same units.

    The comparison is made on the errors: units where both models agree
    (both right or both wrong) say nothing about which one is better, so only
    the units where exactly one model is wrong are used. Under H0 such a unit
    is equally likely to favour either model, which is the sign test applied
    to the disagreements.

    Parameters
    ----------
    first_correct, second_correct : array_like of bool
        Whether each model classified each unit correctly, in the same order.
    """
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Both models must be evaluated on the same units.")
    only_first = int((first & ~second).sum())
    only_second = int((~first & second).sum())
    return McNemarResult(
        only_first_correct=only_first,
        only_second_correct=only_second,
        p_value=binomial_two_sided_p(only_first, only_first + only_second),
    )
