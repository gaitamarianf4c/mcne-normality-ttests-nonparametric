"""Datasets and reference values from the homework statement.

Keeping the data in one place means the notebook and the tests always use
exactly the same numbers (single source of truth). Values are stored as tuples
so they cannot be modified by accident; convert them with ``numpy.asarray``
when needed.

Critical values marked *from the standard table* are not printed in the
statement; they are the usual textbook table values used in the hand solutions.
"""

from __future__ import annotations

# --- Problem 1: response time (ms) of a classification model, 12 requests ---
RESPONSE_TIME_MS: tuple[float, ...] = (
    88, 92, 95, 97, 99, 101, 104, 108, 115, 126, 142, 168,
)

#: Shapiro-Wilk coefficients a_1..a_6 for n = 12 (given).
SHAPIRO_WILK_COEFFICIENTS_N12: tuple[float, ...] = (
    0.5475, 0.3325, 0.2347, 0.1586, 0.0922, 0.0303,
)
#: Critical W for n = 12 (given). H0 is rejected when W < W_critical.
SHAPIRO_WILK_W_CRITICAL_N12: dict[float, float] = {0.01: 0.805, 0.05: 0.859}
#: Critical D for the Lilliefors test, n = 12, alpha = 0.05 (given).
LILLIEFORS_D_CRITICAL_N12: float = 0.242

# --- Problem 2: systolic blood pressure (mmHg), 12 patients ---
SYSTOLIC_BP_MMHG: tuple[float, ...] = (
    118, 125, 131, 122, 128, 135, 119, 127, 133, 124, 129, 126,
)
SYSTOLIC_BP_REFERENCE: float = 120.0
#: t critical, df = 11, alpha = 0.05 two-sided (given).
T_CRITICAL_DF11: float = 2.201
#: Wilcoxon signed-rank critical value, n = 12, alpha = 0.05 two-sided
#: (from the standard table). H0 is rejected when W <= W_critical.
WILCOXON_CRITICAL_N12: float = 13.0

# --- Problem 3: model accuracy (%) on 10 datasets, before/after augmentation ---
ACCURACY_BEFORE: tuple[float, ...] = (
    81.2, 78.5, 85.1, 79.9, 83.4, 77.8, 84.6, 80.3, 82.7, 79.1,
)
ACCURACY_AFTER: tuple[float, ...] = (
    83.0, 79.1, 86.9, 82.4, 83.1, 80.2, 86.0, 82.5, 84.9, 80.0,
)
#: t critical, df = 9, alpha = 0.05 two-sided (given).
T_CRITICAL_DF9: float = 2.262
#: t critical, df = 18, alpha = 0.05 two-sided (from the standard table).
T_CRITICAL_DF18: float = 2.101

# --- Problem 4: hospital stay (days), two treatments, different patients ---
HOSPITAL_STAY_A: tuple[float, ...] = (5, 7, 6, 9, 8, 6, 7, 10, 5, 8)
HOSPITAL_STAY_B: tuple[float, ...] = (8, 10, 9, 12, 7, 11, 13, 9, 10, 14, 8, 11)
#: t critical, df = 20, alpha = 0.05 two-sided (given).
T_CRITICAL_DF20: float = 2.086
#: Mann-Whitney critical U, n1 = 10, n2 = 12, alpha = 0.05 two-sided
#: (from the standard table). H0 is rejected when U <= U_critical.
MANN_WHITNEY_U_CRITICAL_10_12: float = 29.0

#: Two-sided critical z for alpha = 0.05.
Z_CRITICAL_005: float = 1.96
ALPHA: float = 0.05
