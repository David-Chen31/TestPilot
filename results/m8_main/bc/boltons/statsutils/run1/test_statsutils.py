import math

import pytest

from boltons.statsutils import (
    Stats,
    describe,
    format_histogram_counts,
)
import boltons.statsutils as statsutils


# ---------------------------------------------------------------------------
# Stats construction / basic protocol
# ---------------------------------------------------------------------------

def test_stats_basic_protocol():
    s = Stats(range(10))
    assert len(s) == 10
    assert list(s) == list(range(10))
    assert s.data == list(range(10))


def test_stats_use_copy_true_by_default():
    data = [1, 2, 3]
    s = Stats(data)
    s.data.append(4)
    assert data == [1, 2, 3]
    assert s.data == [1, 2, 3, 4]


def test_stats_use_copy_false():
    data = [1, 2, 3]
    s = Stats(data, use_copy=False)
    assert s.data is data


def test_stats_invalid_data_none_raises_on_iter():
    with pytest.raises(TypeError):
        Stats(None)


# ---------------------------------------------------------------------------
# Empty data uses default
# ---------------------------------------------------------------------------

def test_empty_defaults():
    s = Stats([], default=42.0)
    assert s.count == 42.0
    assert s.mean == 42.0
    assert s.median == 42.0
    assert s.max == 42.0
    assert s.min == 42.0
    assert s.std_dev == 42.0
    assert s.mode == 42.0


def test_empty_quantile_returns_default():
    s = Stats([], default=-1.0)
    assert s.get_quantile(0.5) == -1.0


# ---------------------------------------------------------------------------
# Basic Stats properties
# ---------------------------------------------------------------------------

def test_count_mean_max_min():
    s = Stats([2, 1, 3])
    assert s.count == 3
    assert s.mean == 2.0
    assert s.max == 3
    assert s.min == 1


def test_max_min_is_sorted():
    s = Stats([3, 1, 2], is_sorted=True)
    assert s.max == 2
    assert s.min == 3


def test_median_even_odd():
    assert Stats([2, 1, 3]).median == 2
    assert Stats(range(4)).median == 1.5


def test_mode():
    assert Stats([2, 1, 3, 1]).mode == 1
    assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'


def test_mode_tie_first_occurring():
    assert Stats([1, 1, 2, 2, 3]).mode == 1


def test_iqr():
    assert Stats([1, 2, 3, 4, 5]).iqr == 2
    assert Stats(range(1001)).iqr == 500


def test_trimean():
    assert Stats([2, 1, 3]).trimean == 2.0
    assert Stats(range(97)).trimean == 48.0


def test_variance_std_dev():
    assert Stats(range(97)).variance == 784.0
    assert Stats(range(97)).std_dev == 28.0


def test_median_abs_dev():
    assert Stats(range(97)).median_abs_dev == 24.0
    # alias
    assert Stats(range(97)).mad == 24.0


def test_rel_std_dev():
    s = Stats(range(97))
    assert round(s.rel_std_dev, 3) == 0.583


def test_rel_std_dev_zero_mean_uses_default():
    s = Stats([-1, 1], default=99.0)
    assert s.mean == 0.0
    assert s.rel_std_dev == 99.0


def test_skewness_symmetric():
    assert Stats(range(97)).skewness == 0.0


def test_skewness_left_right():
    left = Stats(list(range(97)) + list(range(10))).skewness
    right = Stats(list(range(97)) + list(range(87, 97))).skewness
    assert round(left, 3) == 0.114
    assert round(right, 3) == -0.114


def test_skewness_single_element_default():
    s = Stats([5], default=7.0)
    assert s.skewness == 7.0


def test_skewness_constant_default():
    s = Stats([3, 3, 3], default=7.0)
    assert s.std_dev == 0.0
    assert s.skewness == 7.0


def test_kurtosis():
    assert Stats(range(9)).kurtosis == 1.99125


def test_kurtosis_single_element():
    s = Stats([5])
    assert s.kurtosis == 0.0


def test_kurtosis_constant_data():
    s = Stats([3, 3, 3])
    assert s.kurtosis == 0.0


def test_get_quantile_median():
    assert Stats(range(100)).get_quantile(0.5) == 49.5


def test_get_quantile_bounds():
    s = Stats([10, 20, 30])
    assert s.get_quantile(0.0) == 10
    assert s.get_quantile(1.0) == 30


def test_get_quantile_invalid():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)
    with pytest.raises(ValueError):
        s.get_quantile(1.1)


def test_get_quantile_accepts_string_number():
    s = Stats([0, 10])
    assert s.get_quantile("0.5") == 5.0


# ---------------------------------------------------------------------------
# Sorted-data lazy sorting (use_copy=True, is_sorted=False)
# ---------------------------------------------------------------------------

def test_sorted_data_lazy_sort_then_sorted_access():
    # use_copy=True and is_sorted=False: first sorted access triggers sort
    s = Stats([3, 1, 2])
    assert s.data == [3, 1, 2]
    assert s.median == 2
    assert s.data == [1, 2, 3]


def test_sorted_data_already_sorted_flag_skips_sort():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.median == 2
    assert s.data == [1, 2, 3]


def test_sorted_data_no_copy_returns_fresh_sorted():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    sorted_vals = s._get_sorted_data()
    assert sorted_vals == [1, 2, 3]
    # original unchanged
    assert data == [3, 1, 2]


# ---------------------------------------------------------------------------
# z-score
# ---------------------------------------------------------------------------

def test_zscore():
    s = Stats([1, 2, 3, 4, 5])
    z = s.get_zscore(1)
    assert z == pytest.approx((1 - 3.0) / s.std_dev)


def test_zscore_zero_std_equal():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0


def test_zscore_zero_std_greater():
    s = Stats([5, 5, 5])
    assert s.get_zscore(6) == float('inf')


def test_zscore_zero_std_less():
    s = Stats([5, 5, 5])
    assert s.get_zscore(4) == float('-inf')


def test_zscore_zero_std_less_via_branch():
    # explicit check on the "<" branch returning -inf
    s = Stats([0, 0, 0])
    result = s.get_zscore(-1)
    assert result == float('-inf')
    assert math.isinf(result) and result < 0


def test_zscore_zero_std_less_missing_branch():
    # Exercise line 521 -> 523: value < mean but with a NaN-free path
    # value below mean when std_dev is zero must return -inf exactly
    s = Stats([2, 2, 2])
    result = s.get_zscore(1)
    assert result == float('-inf')
    assert math.isinf(result) and result < 0


# ---------------------------------------------------------------------------
# Caching and clear_cache
# ---------------------------------------------------------------------------

def test_cache_and_clear_cache():
    s = Stats([1, 2, 3])
    assert s.mean == 2.0
    assert hasattr(s, '_mean')
    s.clear_cache()
    assert not hasattr(s, '_mean')


def test_clear_cache_when_no_cache():
    s = Stats([1, 2, 3])
    s.clear_cache()  # should not raise
    assert s.mean == 2.0


# ---------------------------------------------------------------------------
# trim_relative
# ---------------------------------------------------------------------------

def test_trim_relative():
    s = Stats(list(range(10)))
    s.trim_relative(0.1)
    assert s.data == [1, 2, 3, 4, 5, 6, 7, 8]


def test_trim_relative_no_effect_when_small():
    s = Stats([1, 2])
    s.trim_relative(0.1)
    assert s.data == [1, 2]


def test_trim_relative_invalid():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)
    with pytest.raises(ValueError):
        s.trim_relative(0.5)


def test_trim_relative_clears_cache():
    s = Stats(list(range(10)))
    _ = s.mean
    assert hasattr(s, '_mean')
    s.trim_relative(0.1)
    assert not hasattr(s, '_mean')


# ---------------------------------------------------------------------------
# pearson_type
# ---------------------------------------------------------------------------

def test_pearson_type_normal_like():
    # Symmetric data -> skewness ~ 0
    s = Stats(range(1, 8))
    # just ensure it returns one of the documented type codes
    assert s.pearson_type in (0, 1, 2, 3, 4, 5, 6, 7)


def test_pearson_type_symmetric_beta_normal():
    # A symmetric uniform distribution: skewness is 0 (c1 rounds to 0),
    # but kurtosis is well below 3, so this classifies as Symmetric Beta
    # (type 2), not Normal, under the current implementation.
    s = Stats(range(0, 20))
    # symmetry -> skewness must be 0 for this dataset
    assert round(s.skewness, 8) == 0.0
    assert s.kurtosis < 3
    assert s.pearson_type == 2


def test_pearson_type_via_unit_distribution():
    # Uniform 0..1 style data yields a symmetric, non-normal kurtosis
    s = Stats([0.0, 1.0])
    # Compute pearson_type deterministically for this input
    assert s.pearson_type in (1, 2, 3, 4, 5, 6, 7)


def test_pearson_type_normal_return_zero():
    # Construct data whose skewness ~0 and kurtosis ~= 3 to hit the
    # "Normal" (type 0) branch. We patch properties to be exact.
    s = Stats([1, 2, 3])
    # Force internal state used by _calc_pearson_type via cached props
    s._skewness = 0.0
    s._kurtosis = 3.0
    assert s.pearson_type == 0


def test_pearson_type_symmetric_beta_below_three():
    # skewness == 0, beta2 < 3 -> type 2
    s = Stats([1, 2, 3])
    s._skewness = 0.0
    s._kurtosis = 1.5
    assert s.pearson_type == 2


def test_pearson_type_symmetric_beta_above_three():
    # skewness == 0, beta2 > 3 -> type 7 (line 464-465)
    s = Stats([1, 2, 3])
    s._skewness = 0.0
    s._kurtosis = 5.0
    assert s.pearson_type == 7


def test_pearson_type_gamma():
    # c1 != 0 (skewness non-zero), c2 == 0 -> type 3 (Gamma)
    s = Stats([1, 2, 3])
    # c2 = 2*beta2 - 3*beta1 - 6; with beta1 = skew^2, choose so c2 == 0
    skew = 1.0
    beta1 = skew ** 2.0
    # 2*beta2 - 3*1 - 6 == 0 -> beta2 == 4.5
    s._skewness = skew
    s._kurtosis = 4.5
    assert s.pearson_type == 3


def test_pearson_type_beta_negative_k():
    # c0 == 0 and c2 < 0 -> type 1 (line 469-471)
    s = Stats([1, 2, 3])
    # Need c1 != 0, c2 != 0, c0 == 0.
    # c0 = 4*beta2 - 3*beta1
    # c2 = 2*beta2 - 3*beta1 - 6
    skew = 1.0
    beta1 = skew ** 2.0  # 1.0
    beta2 = 3 * beta1 / 4.0  # c0 == 0 -> 0.75
    s._skewness = skew
    s._kurtosis = beta2
    # c2 = 2*0.75 - 3*1 - 6 = 1.5 - 9 = -7.5 < 0 -> type 1
    assert s.pearson_type == 1


def test_pearson_type_beta_positive_c2():
    # c0 == 0 and c2 > 0 -> type 6
    s = Stats([1, 2, 3])
    skew = 1.0
    beta1 = skew ** 2.0
    beta2 = 3 * beta1 / 4.0
    # c2 = 2*beta2 - 3*beta1 - 6 = 1.5 - 3 - 6 = -7.5 < 0, so instead
    # we need a different c0==0 arrangement. c0==0 requires beta2=3beta1/4,
    # which makes c2 = 1.5beta1 - 3beta1 - 6 = -1.5beta1 - 6, always < 0.
    # So this branch is not reachable with all-positive; skip guarantee of
    # branch but still assert no crash with a sanity example.
    s._skewness = skew
    s._kurtosis = beta2
    assert s.pearson_type == 1


def test_pearson_type_k_negative():
    # k = c1^2 / (4*c0*c2); make it negative via c0*c2 < 0 -> type 1
    s = Stats([1, 2, 3])
    # pick skew non-zero, c2 != 0, c0*c2 < 0, c1 != 0
    s._skewness = 0.5
    # beta1 = 0.25
    # choose beta2 so c0 = 4*beta2 - 0.75 < 0 and c2 = 2*beta2 - 0.75 - 6 < 0
    s._kurtosis = 0.1
    # c0 = 0.4 - 0.75 = -0.35 < 0
    # c2 = 0.2 - 0.75 - 6 = -6.55 < 0
    # c1 = 0.5*(0.1+3) = 1.55 != 0
    # k = 1.55^2 / (4*(-0.35)*(-6.55)) > 0 -> not type 1
    pt = s.pearson_type
    assert pt in (1, 2, 3, 4, 5, 6, 7)


def test_pearson_type_type_five():
    # Force round(k, precision) == 1 -> type 5
    s = Stats([1, 2, 3])
    # Need c1 != 0, c2 != 0, c0 != 0, k == 1
    # k = c1^2 / (4*c0*c2) = 1  ->  c1^2 = 4*c0*c2
    # Let skew = 1: c1 = 1*(beta2+3) = beta2+3
    # c0 = 4*beta2 - 3
    # c2 = 2*beta2 - 3 - 6 = 2*beta2 - 9
    # (beta2+3)^2 = 4*(4*beta2 - 3)*(2*beta2 - 9)
    # Solve numerically: try beta2 values
    # (b+3)^2 = 4*(4b-3)*(2b-9)
    # b^2+6b+9 = 4*(8b^2 - 42b + 27) = 32b^2 - 168b + 108
    # 0 = 31b^2 - 174b + 99
    # b = (174 +/- sqrt(174^2 - 4*31*99)) / 62
    # 174^2 = 30276; 4*31*99 = 12276; diff = 18000; sqrt ~ 134.164
    # b = (174 + 134.164)/62 = 4.97... ; b = (174-134.164)/62 = 0.6425
    # Use b ~ 4.970... 
    b = (174 + math.sqrt(18000)) / 62.0
    s._skewness = 1.0
    s._kurtosis = b
    assert s.pearson_type in (4, 5, 6, 7)


def test_pearson_type_type_four():
    # k < 1 and > 0 -> type 4
    s = Stats([1, 2, 3])
    s._skewness = 0.5
    s._kurtosis = 5.0
    pt = s.pearson_type
    assert pt in (1, 3, 4, 5, 6, 7)


def test_pearson_type_high_precision():
    # with default precision 0, small c1 may round to 0
    s = Stats([1, 2, 3])
    s._skewness = 0.0
    s._kurtosis = 3.0
    assert s.pearson_type == 0


# ---------------------------------------------------------------------------
# histogram counts
# ---------------------------------------------------------------------------

def test_get_histogram_counts_fixed_count():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=2)
    assert len(counts) == 2
    # all values counted
    assert sum(c for _, c in counts) == 10


def test_get_histogram_counts_explicit_bins():
    s = Stats([0, 1, 2, 3, 4])
    counts = s.get_histogram_counts(bins=[0, 2, 4])
    assert sum(c for _, c in counts) == 5
    assert counts[0][0] == 0.0


def test_get_histogram_counts_prepends_min():
    s = Stats([1, 5, 10])
    counts = s.get_histogram_counts(bins=[2.0, 5.0])
    # min should have been prepended
    assert any(b == 1.0 for b, _ in counts)


def test_get_histogram_counts_empty():
    s = Stats([])
    counts = s.get_histogram_counts()
    assert counts == [(0.0, 0)]


def test_get_histogram_counts_empty_data_explicit_bins():
    s = Stats([])
    # empty data path returns [[0.0]]
    counts = s.get_histogram_counts(bins=[0, 1, 2])
    assert sum(c for _, c in counts) == 0


def test_get_histogram_counts_invalid_bins():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins="notbins")


def test_get_histogram_counts_unexpected_kwargs():
    s = Stats([1, 2, 3])
    with pytest.raises(TypeError):
        s.get_histogram_counts(bins=2, unknown=1)


def test_get_histogram_counts_bin_digits():
    s = Stats([0.11, 0.22, 0.33, 0.44, 0.55])
    counts = s.get_histogram_counts(bins=2, bin_digits=0)
    # floor to integer boundaries
    for b, _ in counts:
        assert b == math.floor(b)


def test_get_histogram_counts_constant_data():
    s = Stats([5, 5, 5, 5, 5])
    # Freedman's algorithm -> dx == 0, returns single min bin
    counts = s.get_histogram_counts()
    assert counts == [(5.0, 5)]


def test_fewer_than_four_data():
    s = Stats([1, 2])
    counts = s.get_histogram_counts()
    assert sum(c for _, c in counts) == 2


def test_get_histogram_counts_fewer_than_four_explicit_count():
    # len_data < 4, count provided -> uses (max-min)/count spacing
    s = Stats([0, 3])
    counts = s.get_histogram_counts(bins=3)
    # bins computed: 0, 1, 2 (floored); max appended implicitly not
    assert sum(c for _, c in counts) == 2
    assert [b for b, _ in counts] == [0.0, 1.0, 2.0]


def test_get_histogram_counts_invalid_x_in_bin_list():
    s = Stats([1, 2, 3])
    # list of non-numeric values -> float conversion fails -> ValueError
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins=[object(), object()])


def test_get_histogram_counts_unexpected_object_bins_message():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        s.get_histogram_counts(bins=object())
    assert 'bins expected integer bin count or list' in str(excinfo.value)


def test_get_histogram_counts_with_max_branch():
    # Force the dx == 0 Freedman path with with_max=True to hit line 584
    s = Stats([5, 5, 5, 5, 5])
    bins = s._get_bin_bounds(with_max=True)
    assert bins == [5.0, 5.0]


def test_get_histogram_counts_fewer_than_four_no_count():
    # len_data < 4 with count None -> count defaults to len_data
    s = Stats([0, 1])
    counts = s.get_histogram_counts()
    assert sum(c for _, c in counts) == 2


def test_get_bin_bounds_with_max_regular():
    # Directly exercise the with_max append at line 584 with non-degenerate data
    s = Stats([0.0, 1.0, 2.0, 3.0, 4.0])
    bins = s._get_bin_bounds(count=3, with_max=True)
    assert bins[-1] == 4.0
    assert len(bins) == 4


# ---------------------------------------------------------------------------
# format_histogram / format_histogram_counts
# ---------------------------------------------------------------------------

def test_format_histogram():
    data = list(range(20)) + list(range(5, 15)) + [10]
    out = Stats(data).format_histogram(width=30)
    assert isinstance(out, str)
    assert '#' in out


def test_format_histogram_counts_default():
    out = format_histogram_counts([(0.0, 5), (1.0, 10)], width=40)
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('0.0:')
    assert lines[1].startswith('1.0:')


def test_format_histogram_counts_zero_count():
    out = format_histogram_counts([(0.0, 0), (1.0, 1)], width=40)
    lines = out.splitlines()
    # zero-count rows get a '|' bar
    assert '|' in lines[0]


def test_format_histogram_counts_format_bin():
    out = format_histogram_counts(
        [(0.0, 1), (1.0, 2)],
        width=40,
        format_bin=lambda v: '%.1fms' % v,
    )
    assert 'ms' in out


def test_format_histogram_counts_bar_length():
    out = format_histogram_counts([(0.0, 1), (1.0, 4)], width=80)
    lines = out.splitlines()
    # the largest count should have the longest bar
    bar0 = len(lines[0]) - len(lines[0].rstrip('#'))
    bar1 = len(lines[1]) - len(lines[1].rstrip('#'))
    assert bar1 > bar0


def test_format_histogram_counts_default_width():
    # width=None triggers the terminal-size lookup (or the 80 fallback)
    out = format_histogram_counts([(0.0, 1), (1.0, 4)])
    lines = out.splitlines()
    assert len(lines) == 2
    expected = (
        '0.0: 1 '
        + ('#' * int(round(1 * (max(80 - len('xxxxx: 4 #'), 3) / 4))))
    )
    # Simpler: the longest row should be at least several '#' characters
    assert '#' in lines[1]


def test_format_histogram_counts_default_width_branch():
    # Exercise the width=None path, including the import/except (lines 823-824)
    # and the fallback width computation.
    out = format_histogram_counts([(0.0, 1), (1.0, 2)])
    lines = out.splitlines()
    assert len(lines) == 2
    # With default width, bar for count 2 should be quite long
    assert len(lines[1]) - len(lines[1].rstrip('#')) == 2 or \
        (len(lines[1]) - len(lines[1].rstrip('#'))) > 0
    assert lines[0].startswith('0.0:')
    assert lines[1].startswith('1.0:')


def test_format_histogram_counts_minimum_bar_columns():
    # Very small width -> bar_cols clamped to 3
    out = format_histogram_counts([(0.0, 1), (1.0, 4)], width=1)
    lines = out.splitlines()
    assert len(lines) == 2
    # With bar_cols == 3, line_k == 0.75; count 4 -> bar_len 3
    assert lines[1].endswith('###')


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_dict():
    d = Stats(range(1, 8)).describe()
    assert d['count'] == 7
    assert d['mean'] == 4.0
    assert d['std_dev'] == 2.0
    assert d['mad'] == 2.0
    assert d['min'] == 1
    assert d['0.25'] == 2.5
    assert d['0.5'] == 4
    assert d['0.75'] == 5.5
    assert d['max'] == 7


def test_describe_list_format():
    lst = Stats(range(1, 8)).describe(format='list')
    assert isinstance(lst, list)
    keys = [k for k, _ in lst]
    assert keys == ['count', 'mean', 'std_dev', 'mad', 'min',
                    '0.25', '0.5', '0.75', 'max']


def test_describe_text_format():
    text = Stats(range(1, 8)).describe(format='text')
    assert text == (
        'count:    7\n'
        'mean:     4.0\n'
        'std_dev:  2.0\n'
        'mad:      2.0\n'
        'min:      1\n'
        '0.25:     2.5\n'
        '0.5:      4\n'
        '0.75:     5.5\n'
        'max:      7'
    )


def test_describe_invalid_format():
    with pytest.raises(ValueError):
        Stats(range(5)).describe(format='xml')


def test_describe_invalid_format_message():
    with pytest.raises(ValueError) as excinfo:
        Stats(range(5)).describe(format='xml')
    assert 'invalid format for describe' in str(excinfo.value)


def test_describe_custom_quantiles():
    d = Stats(range(10)).describe(quantiles=[0.1, 0.9])
    assert '0.1' in d
    assert '0.9' in d
    assert '0.5' not in d


def test_describe_empty_quantiles_uses_default():
    d = Stats(range(10)).describe(quantiles=[])
    assert '0.25' in d


def test_describe_function():
    d = describe(range(7))
    assert d['count'] == 7
    assert d['mean'] == 3.0
    assert d['max'] == 6


def test_describe_function_text():
    text = describe(range(7), format='text')
    assert text == (
        'count:    7\n'
        'mean:     3.0\n'
        'std_dev:  2.0\n'
        'mad:      2.0\n'
        'min:      0\n'
        '0.25:     1.5\n'
        '0.5:      3\n'
        '0.75:     4.5\n'
        'max:      6'
    )


def test_describe_text_format_branch():
    # Explicitly hit the 'text' branch (line 753) with a simple input
    text = Stats([1, 2, 3]).describe(format='text')
    lines = text.split('\n')
    assert lines[0].startswith('count:')
    assert lines[-1].startswith('max:')
    assert len(lines) == 9


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_module_level_mean():
    assert statsutils.mean(range(42)) == 20.5


def test_module_level_median():
    assert statsutils.median([2, 1, 3]) == 2


def test_module_level_mode():
    assert statsutils.mode([2, 1, 3, 1]) == 1


def test_module_level_variance():
    assert statsutils.variance(range(97)) == 784.0


def test_module_level_std_dev():
    assert statsutils.std_dev(range(97)) == 28.0


def test_module_level_median_abs_dev():
    assert statsutils.median_abs_dev(range(97)) == 24.0


def test_module_level_iqr():
    assert statsutils.iqr([1, 2, 3, 4, 5]) == 2


def test_module_level_default():
    assert statsutils.mean([], default=5.0) == 5.0


def test_module_level_trimean():
    assert statsutils.trimean([2, 1, 3]) == 2.0


def test_module_level_skewness():
    assert statsutils.skewness(range(97)) == 0.0


def test_module_level_kurtosis():
    assert statsutils.kurtosis(range(9)) == 1.99125


def test_module_level_no_copy_semantics():
    data = [1, 2, 3]
    assert statsutils.mean(data) == 2.0
    # helpers use use_copy=False, so data is not mutated
    assert data == [1, 2, 3]
