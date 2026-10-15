"""Manual implementations of the statistical tests from *Teste statistice - Curs 2*.

Every function reproduces, step by step, a calculation that is also done by
hand on paper, with the formulas used in the course. The intermediate values
are returned (not printed) so that the notebook can display them and compare
the results with ``scipy.stats``.

Modules
-------
data
    The datasets and the reference values (critical values, coefficients)
    given in the homework statement.
descriptive
    Mean, variance, standard deviation, standard error.
ranks
    Average ranks for equal values, shared by the rank-based tests.
normality
    Part 1 of the course: Lilliefors table and Shapiro-Wilk W.
one_sample
    Part 2: one-sample t-test, sign test, Wilcoxon signed-rank test.
two_sample
    Parts 3 and 4: paired t-test, independent t-test, Mann-Whitney;
    McNemar for two models evaluated on the same images.
"""

from curs2_stats.descriptive import Descriptives, describe
from curs2_stats.normality import (
    LillieforsTable,
    ShapiroWilkSteps,
    lilliefors_table,
    shapiro_wilk_w,
)
from curs2_stats.one_sample import (
    SignTestResult,
    TTestResult,
    WilcoxonResult,
    one_sample_t,
    sign_test,
    wilcoxon_signed_rank,
)
from curs2_stats.ranks import average_ranks
from curs2_stats.two_sample import (
    IndependentTResult,
    MannWhitneyResult,
    McNemarResult,
    independent_t,
    mann_whitney,
    mcnemar_exact,
    paired_t,
)

__all__ = [
    "Descriptives",
    "IndependentTResult",
    "LillieforsTable",
    "MannWhitneyResult",
    "McNemarResult",
    "ShapiroWilkSteps",
    "SignTestResult",
    "TTestResult",
    "WilcoxonResult",
    "average_ranks",
    "describe",
    "independent_t",
    "lilliefors_table",
    "mann_whitney",
    "mcnemar_exact",
    "one_sample_t",
    "paired_t",
    "shapiro_wilk_w",
    "sign_test",
    "wilcoxon_signed_rank",
]

__version__ = "1.0.0"
