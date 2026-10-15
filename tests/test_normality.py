"""Problem 1: Lilliefors table and Shapiro-Wilk W."""

import numpy as np
import pytest
from scipy import stats

from curs2_stats import data, lilliefors_table, shapiro_wilk_w


@pytest.fixture
def sample():
    return np.asarray(data.RESPONSE_TIME_MS, dtype=float)


def test_lilliefors_d_matches_ks_on_standardised_data(sample):
    z = (sample - sample.mean()) / sample.std(ddof=1)
    expected = stats.kstest(z, "norm").statistic
    assert lilliefors_table(sample).statistic == pytest.approx(expected)


def test_lilliefors_hand_value_and_decision(sample):
    table = lilliefors_table(sample)
    assert table.statistic == pytest.approx(0.222, abs=5e-4)
    assert table.statistic == table.d_plus_max > table.d_minus_max
    assert table.argmax == 108
    assert not table.reject(data.LILLIEFORS_D_CRITICAL_N12)


def test_lilliefors_d_is_a_probability_distance(sample):
    assert 0.0 <= lilliefors_table(sample).statistic <= 1.0


def test_shapiro_wilk_matches_scipy(sample):
    steps = shapiro_wilk_w(sample, data.SHAPIRO_WILK_COEFFICIENTS_N12)
    # SciPy approximates the coefficients (Royston), so allow a small gap.
    assert steps.statistic == pytest.approx(stats.shapiro(sample).statistic, abs=1e-3)


def test_shapiro_wilk_hand_value_and_decision(sample):
    steps = shapiro_wilk_w(sample, data.SHAPIRO_WILK_COEFFICIENTS_N12)
    assert steps.statistic == pytest.approx(0.838, abs=5e-4)
    assert steps.reject(data.SHAPIRO_WILK_W_CRITICAL_N12[0.05])
    assert not steps.reject(data.SHAPIRO_WILK_W_CRITICAL_N12[0.01])


def test_shapiro_wilk_checks_coefficient_count(sample):
    with pytest.raises(ValueError):
        shapiro_wilk_w(sample, [0.5, 0.3])
