"""Problem 2: one-sample t-test, sign test, Wilcoxon signed-rank."""

import numpy as np
import pytest
from scipy import stats

from curs2_stats import data, one_sample_t, sign_test, wilcoxon_signed_rank

BP = np.asarray(data.SYSTOLIC_BP_MMHG, dtype=float)
MU0 = data.SYSTOLIC_BP_REFERENCE


def test_one_sample_t_matches_scipy():
    result = one_sample_t(BP, MU0)
    expected = stats.ttest_1samp(BP, MU0)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.df == expected.df
    assert result.reject(data.T_CRITICAL_DF11)


def test_one_sample_confidence_interval_matches_scipy():
    result = one_sample_t(BP, MU0)
    low, high = result.confidence_interval(data.T_CRITICAL_DF11)
    expected = stats.ttest_1samp(BP, MU0).confidence_interval()
    # The interval of the mean is the interval of (mean - mu0) shifted by mu0.
    assert MU0 + low == pytest.approx(expected.low, abs=1e-3)
    assert MU0 + high == pytest.approx(expected.high, abs=1e-3)


def test_sign_test_matches_scipy_binomtest():
    result = sign_test(BP, MU0)
    assert (result.n_plus, result.n_minus, result.n_ties) == (10, 2, 0)
    expected = stats.binomtest(result.n_plus, result.n, 0.5).pvalue
    assert result.p_value == pytest.approx(expected)
    assert result.reject(data.ALPHA)


def test_sign_test_normal_approximation():
    # z = (10 - 6) / sqrt(3)
    assert sign_test(BP, MU0).z == pytest.approx(4 / np.sqrt(3))


def test_wilcoxon_rank_sums_and_decision():
    result = wilcoxon_signed_rank(BP, MU0)
    assert result.w_minus == pytest.approx(3.5)
    assert result.w_plus == pytest.approx(74.5)
    assert result.w_plus + result.w_minus == pytest.approx(result.n * (result.n + 1) / 2)
    assert result.reject(data.WILCOXON_CRITICAL_N12)


def test_wilcoxon_statistic_matches_scipy():
    result = wilcoxon_signed_rank(BP, MU0)
    assert result.statistic == pytest.approx(stats.wilcoxon(BP - MU0).statistic)


def test_wilcoxon_normal_approximation_course_formula():
    # mu = 12*13/4 = 39, sigma = sqrt(12*13*25/24) = sqrt(162.5)
    result = wilcoxon_signed_rank(BP, MU0)
    assert result.z == pytest.approx((74.5 - 39) / np.sqrt(162.5))
