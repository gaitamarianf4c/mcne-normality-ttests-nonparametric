"""Descriptive statistics and ranking against NumPy/SciPy."""

import numpy as np
import pytest
from scipy import stats

from curs2_stats import average_ranks, describe
from curs2_stats import data


def test_describe_matches_numpy():
    x = np.asarray(data.RESPONSE_TIME_MS)
    result = describe(x)
    assert result.n == 12
    assert result.mean == pytest.approx(111.25)
    assert result.variance == pytest.approx(np.var(x, ddof=1))
    assert result.std == pytest.approx(np.std(x, ddof=1))
    assert result.standard_error == pytest.approx(stats.sem(x))


def test_describe_rejects_single_value():
    with pytest.raises(ValueError):
        describe([5.0])


def test_average_ranks_matches_scipy_with_ties():
    combined = np.concatenate([data.HOSPITAL_STAY_A, data.HOSPITAL_STAY_B])
    np.testing.assert_allclose(average_ranks(combined), stats.rankdata(combined))


def test_rank_sum_identity():
    combined = np.concatenate([data.HOSPITAL_STAY_A, data.HOSPITAL_STAY_B])
    n = combined.size
    assert average_ranks(combined).sum() == pytest.approx(n * (n + 1) / 2)
