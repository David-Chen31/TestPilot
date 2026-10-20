"""配对比较统计量。"""

import pytest

from testpilot.eval.stats import bootstrap_ci, compare, paired_diffs, sign_flip_p_value


def test_exact_sign_flip_p_value():
    # 3 个目标全部 +1：2^3 种符号中只有全正 / 全负两种不小于观测值
    assert sign_flip_p_value([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)
    assert sign_flip_p_value([0.0, 0.0]) == 1.0
    assert sign_flip_p_value([1.0, -1.0]) == 1.0


def test_monte_carlo_p_value_never_zero():
    p = sign_flip_p_value([1.0] * 20)
    assert 0 < p < 0.001


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = [0.01, 0.03, 0.05, 0.02, 0.04]
    low, high = bootstrap_ci(values)
    assert (low, high) == bootstrap_ci(values)  # 固定种子
    assert low < 0.03 < high
    assert bootstrap_ci([0.2, 0.2]) == (0.2, 0.2)


def test_compare_pairs_by_target_and_averages_runs_within_target():
    a = {"t1": [0.5, 0.7], "t2": [0.9], "only_a": [0.1]}
    b = {"t1": [0.8], "t2": [0.9, 0.9], "only_b": [1.0]}
    assert paired_diffs(a, b) == [(pytest.approx(0.6), 0.8), (0.9, 0.9)]
    c = compare(a, b)
    assert c.n_targets == 2 and (c.wins, c.losses, c.ties) == (1, 0, 1)
    assert c.mean_diff == pytest.approx(0.1)
    with pytest.raises(ValueError):
        compare({"x": [1.0]}, {"y": [1.0]})
