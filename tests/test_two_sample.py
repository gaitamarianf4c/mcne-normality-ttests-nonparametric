"""Problems 3, 4 and 5.4: paired, independent and rank-based comparisons."""

import numpy as np
import pytest
from scipy import stats

from curs2_stats import data, independent_t, mann_whitney, mcnemar_exact, paired_t

BEFORE = np.asarray(data.ACCURACY_BEFORE, dtype=float)
AFTER = np.asarray(data.ACCURACY_AFTER, dtype=float)
A = np.asarray(data.HOSPITAL_STAY_A, dtype=float)
B = np.asarray(data.HOSPITAL_STAY_B, dtype=float)


def test_paired_t_matches_scipy():
    result = paired_t(AFTER, BEFORE)
    expected = stats.ttest_rel(AFTER, BEFORE)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.df == expected.df
    assert result.reject(data.T_CRITICAL_DF9)


def test_paired_t_confidence_interval_matches_scipy():
    low, high = paired_t(AFTER, BEFORE).confidence_interval(data.T_CRITICAL_DF9)
    expected = stats.ttest_rel(AFTER, BEFORE).confidence_interval()
    assert (low, high) == (pytest.approx(expected.low, abs=1e-3), pytest.approx(expected.high, abs=1e-3))


def test_paired_t_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_t(AFTER, BEFORE[:-1])


def test_same_data_as_independent_loses_significance():
    paired = paired_t(AFTER, BEFORE)
    independent = independent_t(AFTER, BEFORE)
    assert independent.estimate == pytest.approx(paired.estimate)
    assert independent.standard_error > 4 * paired.standard_error
    assert not independent.reject(data.T_CRITICAL_DF18)


def test_independent_t_matches_scipy():
    result = independent_t(A, B)
    expected = stats.ttest_ind(A, B)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.df == expected.df
    assert result.reject(data.T_CRITICAL_DF20)


def test_independent_t_variance_rule_of_thumb():
    result = independent_t(A, B)
    assert result.std_ratio == pytest.approx(np.std(B, ddof=1) / np.std(A, ddof=1))
    assert result.std_ratio < 2
    assert result.variance_ratio == pytest.approx(result.std_ratio**2)


def test_confidence_interval_and_effect_size():
    result = independent_t(B, A)
    low, high = result.confidence_interval(data.T_CRITICAL_DF20)
    assert (low, high) == (pytest.approx(1.34, abs=0.01), pytest.approx(4.79, abs=0.01))
    assert result.cohen_d == pytest.approx(1.59, abs=0.01)


def test_mann_whitney_hand_values():
    result = mann_whitney(A, B)
    assert (result.rank_sum_a, result.rank_sum_b) == (70, 183)
    assert (result.u_a, result.u_b) == (15, 105)
    assert result.u_a + result.u_b == A.size * B.size
    assert result.reject(data.MANN_WHITNEY_U_CRITICAL_10_12)


def test_mann_whitney_statistic_matches_scipy():
    assert mann_whitney(A, B).statistic == pytest.approx(stats.mannwhitneyu(A, B).statistic)


def test_mann_whitney_normal_approximation_course_formula():
    # T = 70 (smaller group A), mu = 10*23/2 = 115, sigma = sqrt(10*12*23/12)
    assert mann_whitney(A, B).z == pytest.approx((70 - 115) / np.sqrt(230))


def test_mcnemar_uses_only_disagreements():
    first = np.array([1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    second = np.array([1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    result = mcnemar_exact(first, second)
    assert (result.only_first_correct, result.only_second_correct) == (3, 1)
    assert result.p_value == pytest.approx(stats.binomtest(3, 4, 0.5).pvalue)
