import math

import pytest

from boltons.statsutils import (
    Stats,
    describe,
    format_histogram_counts,
    mean,
    median,
    mode,
    variance,
    std_dev,
    median_abs_dev,
    rel_std_dev,
    skewness,
    kurtosis,
    trimean,
    iqr,
    pearson_type,
)


# ---------------------------------------------------------------------------
# Stats: construction and basic protocol
# ---------------------------------------------------------------------------

def test_stats_copies_data_by_default():
    data = [3, 1, 2]
    s = Stats(data)
    assert s.data == [3, 1, 2]
    assert s.data is not data


def test_stats_no_copy():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    assert s.data is data


def test_stats_len_and_iter():
    s = Stats([4, 5, 6])
    assert len(s) == 3
    assert list(s) == [4, 5, 6]


def test_stats_default_value():
    s = Stats([], default=float('nan'))
    assert math.isnan(s.mean)
    assert math.isnan(s.mean)  # default is cached
    assert s.default is s.mean or math.isnan(s.mean)


def test_stats_default_zero():
    s = Stats([])
    assert s.default == 0.0
    assert s.mean == 0.0
    assert s.median == 0.0
    assert s.max == 0.0
    assert s.min == 0.0
    assert s.mode == 0.0


# ---------------------------------------------------------------------------
# Stats: basic measures
# ---------------------------------------------------------------------------

def test_stats_count_mean_max_min():
    s = Stats(range(20))
    assert s.count == 20
    assert s.mean == 9.5
    assert s.max == 19
    assert s.min == 0


def test_stats_sorted_max_min():
    s = Stats([2, 1, 3], is_sorted=True)
    assert s.max == 3
    assert s.min == 2  # assumes first element is minimum


def test_stats_median():
    assert Stats([2, 1, 3]).median == 2
    assert Stats(range(97)).median == 48
    assert Stats([1, 2, 3, 4]).median == 2.5


def test_stats_mode():
    assert Stats([2, 1, 3, 1]).mode == 1
    assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'
    assert Stats([1, 1, 2, 2, 3]).mode == 1


def test_stats_iqr():
    assert Stats([1, 2, 3, 4, 5]).iqr == 2
    assert Stats(range(1001)).iqr == 500


def test_stats_trimean():
    assert Stats([2, 1, 3]).trimean == 2.0
    assert Stats(range(97)).trimean == 48.0


def test_stats_variance_std_dev():
    s = Stats(range(97))
    assert s.variance == 784.0
    assert s.std_dev == 28.0


def test_stats_median_abs_dev():
    assert Stats(range(97)).median_abs_dev == 24.0
    assert Stats(range(97)).mad == 24.0


def test_stats_rel_std_dev():
    assert round(Stats(range(97)).rel_std_dev, 3) == 0.583


def test_stats_rel_std_dev_zero_mean():
    s = Stats([], default=42.0)
    assert s.rel_std_dev == 42.0


def test_stats_rel_std_dev_zero_mean_data():
    # data [-1, 1] has mean 0 -> rel_std_dev returns default
    s = Stats([-1, 1], default=7.0)
    assert s.rel_std_dev == 7.0


def test_stats_skewness():
    assert skewness(range(97)) == 0.0
    left = skewness(list(range(97)) + list(range(10)))
    right = skewness(list(range(97)) + list(range(87, 97)))
    assert round(left, 3) == 0.114
    assert round(right, 3) == -0.114


def test_stats_skewness_single_value():
    s = Stats([5], default=-1.0)
    assert s.skewness == -1.0


def test_stats_skewness_equal_values():
    s = Stats([5, 5, 5], default=-1.0)
    assert s.skewness == -1.0


def test_stats_kurtosis():
    assert Stats(range(9)).kurtosis == 1.99125


def test_stats_kurtosis_single():
    assert Stats([5]).kurtosis == 0.0


def test_stats_kurtosis_equal():
    assert Stats([5, 5, 5]).kurtosis == 0.0


# ---------------------------------------------------------------------------
# Stats: caching and clear_cache
# ---------------------------------------------------------------------------

def test_stats_caching():
    s = Stats([1, 2, 3])
    assert s._mean == s.mean
    assert s._mean == s.mean


def test_clear_cache():
    s = Stats([1, 2, 3])
    _ = s.mean
    assert hasattr(s, '_mean')
    s.clear_cache()
    assert not hasattr(s, '_mean')


def test_clear_cache_after_adding_data():
    s = Stats([1, 2, 3])
    first_mean = s.mean
    s.data.append(100)
    s.clear_cache()
    assert s.mean != first_mean


# ---------------------------------------------------------------------------
# Stats: quantiles and z-scores
# ---------------------------------------------------------------------------

def test_get_quantile():
    assert Stats(range(100)).get_quantile(0.5) == 49.5
    assert Stats(range(100)).get_quantile(0.0) == 0
    assert Stats(range(100)).get_quantile(1.0) == 99


@pytest.mark.parametrize('q', [-0.1, 1.1, 2])
def test_get_quantile_invalid(q):
    with pytest.raises(ValueError):
        Stats([1, 2, 3]).get_quantile(q)


def test_get_quantile_empty():
    s = Stats([], default=99.0)
    assert s.get_quantile(0.5) == 99.0


def test_get_zscore():
    s = Stats([0, 2, 4])  # mean 2, std_dev ~1.633
    z = s.get_zscore(2)
    assert round(z, 6) == 0.0
    assert s.get_zscore(4) > 0
    assert s.get_zscore(0) < 0


def test_get_zscore_zero_std_dev():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0
    assert s.get_zscore(6) == float('inf')
    assert s.get_zscore(4) == float('-inf')


# ---------------------------------------------------------------------------
# Stats: trimming
# ---------------------------------------------------------------------------

def test_trim_relative():
    s = Stats(range(10))
    s.trim_relative(0.2)
    assert s.data == [2, 3, 4, 5, 6, 7]


def test_trim_relative_modifies_in_place_and_clears_cache():
    s = Stats(range(10))
    _ = s.mean
    s.trim_relative(0.2)
    assert not hasattr(s, '_mean')


def test_trim_relative_noop_small():
    s = Stats([1, 2, 3])
    s.trim_relative(0.1)  # int(3*0.1) == 0 -> no change
    assert s.data == [1, 2, 3]


@pytest.mark.parametrize('amount', [-0.1, 0.5, 1.0])
def test_trim_relative_invalid(amount):
    with pytest.raises(ValueError):
        Stats([1, 2, 3]).trim_relative(amount)


# ---------------------------------------------------------------------------
# Stats: histogram counts
# ---------------------------------------------------------------------------

def test_get_histogram_counts_empty():
    s = Stats([])
    assert s.get_histogram_counts() == [(0.0, 0)]


def test_get_histogram_counts_fixed_count():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=2)
    total = sum(c for _, c in counts)
    assert total == 10
    assert len(counts) == 2


def test_get_histogram_counts_bounds():
    s = Stats([1.0, 2.0, 3.0])
    counts = s.get_histogram_counts(bins=[1.0, 2.0, 3.0])
    assert sum(c for _, c in counts) == 3


def test_get_histogram_counts_bounds_prepended_min():
    s = Stats([0.0, 1.0, 2.0])
    counts = s.get_histogram_counts(bins=[1.0, 2.0])
    # min (0.0) prepended because it is below first boundary
    assert counts[0][0] == 0.0


def test_get_histogram_counts_bad_bins():
    with pytest.raises(ValueError):
        Stats([1, 2, 3]).get_histogram_counts(bins=object())


def test_get_histogram_counts_bad_kw():
    with pytest.raises(TypeError):
        Stats([1, 2, 3]).get_histogram_counts(bogus=1)


def test_get_histogram_counts_default_freedman():
    data = list(range(20)) + list(range(5, 15)) + [10]
    counts = Stats(data).get_histogram_counts()
    assert sum(c for _, c in counts) == len(data)


def test_get_histogram_counts_dx_zero_with_max():
    s = Stats([5.0, 5.0, 5.0, 5.0])
    counts = s.get_histogram_counts()
    assert counts == [(5.0, 4)]


# ---------------------------------------------------------------------------
# Stats: format_histogram and describe
# ---------------------------------------------------------------------------

def test_format_histogram_docstring_example():
    data = list(range(20)) + list(range(5, 15)) + [10]
    result = Stats(data).format_histogram(width=30)
    assert result == (
        " 0.0:  5 #########\n"
        " 4.4:  8 ###############\n"
        " 8.9: 11 ####################\n"
        "13.3:  5 #########\n"
        "17.8:  2 ####"
    )


def test_describe_text():
    stats = Stats(range(1, 8))
    out = stats.describe(format='text')
    assert out == (
        "count:    7\n"
        "mean:     4.0\n"
        "std_dev:  2.0\n"
        "mad:      2.0\n"
        "min:      1\n"
        "0.25:     2.5\n"
        "0.5:      4\n"
        "0.75:     5.5\n"
        "max:      7"
    )


def test_describe_dict():
    result = Stats(range(1, 8)).describe()
    assert result['count'] == 7
    assert result['mean'] == 4.0
    assert result['min'] == 1
    assert result['max'] == 7
    assert result['0.25'] == 2.5
    assert result['0.5'] == 4
    assert result['0.75'] == 5.5


def test_describe_list():
    result = Stats(range(1, 8)).describe(format='list')
    assert isinstance(result, list)
    assert result[0] == ('count', 7)
    assert result[-1] == ('max', 7)


def test_describe_custom_quantiles():
    result = Stats(range(1, 8)).describe(quantiles=[0.1, 0.9], format='dict')
    assert '0.1' in result
    assert '0.9' in result
    assert '0.25' not in result


def test_describe_invalid_format():
    with pytest.raises(ValueError):
        Stats([1, 2, 3]).describe(format='json')


# ---------------------------------------------------------------------------
# describe module-level function
# ---------------------------------------------------------------------------

def test_describe_module_function():
    result = describe(range(7), format='dict')
    assert result['count'] == 7
    assert result['mean'] == 3.0
    assert result['min'] == 0
    assert result['max'] == 6


def test_describe_module_function_text():
    out = describe(range(7), format='text')
    assert out.startswith('count:    7')


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_mean_function():
    assert mean(range(20)) == 9.5
    assert mean(list(range(19)) + [949]) == 56.0


def test_median_function():
    assert median([2, 1, 3]) == 2
    assert median(range(97)) == 48


def test_mode_function():
    assert mode([2, 1, 3, 1]) == 1
    assert mode(['a', 'b', 'b', 'c', 'c', 'c']) == 'c'


def test_variance_function():
    assert variance(range(97)) == 784.0


def test_std_dev_function():
    assert std_dev(range(97)) == 28.0


def test_median_abs_dev_function():
    assert median_abs_dev(range(97)) == 24.0


def test_rel_std_dev_function():
    assert round(rel_std_dev(range(97)), 3) == 0.583


def test_skewness_function():
    assert skewness(range(97)) == 0.0


def test_kurtosis_function():
    assert kurtosis(range(9)) == 1.99125


def test_trimean_function():
    assert trimean([2, 1, 3]) == 2.0


def test_iqr_function():
    assert iqr([1, 2, 3, 4, 5]) == 2


def test_pearson_type_function():
    # Just assert it returns an int for a regular dataset
    result = pearson_type(range(100))
    assert isinstance(result, int)


def test_convenience_function_default():
    assert mean([], default=99.0) == 99.0


# ---------------------------------------------------------------------------
# format_histogram_counts
# ---------------------------------------------------------------------------

def test_format_histogram_counts_basic():
    result = format_histogram_counts([(0.0, 2), (1.0, 4)], width=30)
    lines = result.split('\n')
    assert len(lines) == 2
    assert lines[0].startswith('0.0: 2 ')
    assert lines[1].startswith('1.0: 4 ')


def test_format_histogram_counts_format_bin():
    result = format_histogram_counts([(0.0, 1)], width=20,
                                     format_bin=lambda v: '%dms' % v)
    assert result.startswith('0ms: 1')


def test_format_histogram_counts_zero_count_bar():
    result = format_histogram_counts([(0.0, 0)], width=20)
    assert '|' in result


def test_format_histogram_counts_width_too_small():
    # bar_cols clamped to minimum 3
    result = format_histogram_counts([(0.0, 10)], width=1)
    assert result.count('#') >= 3
