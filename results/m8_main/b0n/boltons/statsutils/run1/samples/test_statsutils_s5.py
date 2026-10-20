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
# Stats basics: construction, len, iter
# ---------------------------------------------------------------------------

def test_stats_len_and_iter():
    s = Stats([1, 2, 3, 4])
    assert len(s) == 4
    assert list(s) == [1, 2, 3, 4]


def test_stats_copies_data_by_default():
    data = [3, 1, 2]
    s = Stats(data)
    assert s.data == [3, 1, 2]
    assert s.data is not data
    # mutating original doesn't affect stats
    data.append(100)
    assert s.data == [3, 1, 2]


def test_stats_use_copy_false_shares_data():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    assert s.data is data


def test_stats_default_value():
    s = Stats([], default=float('nan'))
    assert math.isnan(s.default)
    assert math.isnan(s.mean)


# ---------------------------------------------------------------------------
# Basic Stats properties
# ---------------------------------------------------------------------------

def test_count():
    assert Stats(range(20)).count == 20


def test_mean_property():
    assert Stats(range(42)).mean == 20.5


def test_min_max_property():
    s = Stats([2, 1, 3])
    assert s.min == 1
    assert s.max == 3


def test_min_max_sorted():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.min == 1
    assert s.max == 3


def test_median_property():
    assert Stats([2, 1, 3]).median == 2
    assert Stats(range(97)).median == 48


def test_mode_property():
    assert Stats([2, 1, 3, 1]).mode == 1
    assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'
    assert Stats([1, 1, 2, 2, 3]).mode == 1


def test_iqr_property():
    assert Stats([1, 2, 3, 4, 5]).iqr == 2
    assert Stats(range(1001)).iqr == 500


def test_trimean_property():
    assert Stats([2, 1, 3]).trimean == 2.0
    assert Stats(range(97)).trimean == 48.0


def test_variance_property():
    assert Stats(range(97)).variance == 784.0


def test_std_dev_property():
    assert Stats(range(97)).std_dev == 28.0


def test_median_abs_dev_property():
    assert Stats(range(97)).median_abs_dev == 24.0
    # mad alias
    assert Stats(range(97)).mad == 24.0


def test_rel_std_dev_property():
    s = Stats(range(97))
    assert abs(s.rel_std_dev - (28.0 / 48.0)) < 1e-12


def test_rel_std_dev_zero_mean_returns_default():
    s = Stats([-1, 1], default=123.0)
    assert s.mean == 0.0
    assert s.rel_std_dev == 123.0


def test_skewness_property():
    assert Stats(range(97)).skewness == 0.0


def test_skewness_default_when_single_value():
    s = Stats([5], default=-1.0)
    assert s.skewness == -1.0


def test_kurtosis_property():
    assert Stats(range(9)).kurtosis == 1.99125


def test_kurtosis_single_value_zero():
    assert Stats([5]).kurtosis == 0.0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_module_level_mean():
    assert mean(range(42)) == 20.5


def test_module_level_median():
    assert median([2, 1, 3]) == 2
    assert median(range(97)) == 48


def test_module_level_mode():
    assert mode([2, 1, 3, 1]) == 1


def test_module_level_variance_std_dev():
    assert variance(range(97)) == 784.0
    assert std_dev(range(97)) == 28.0


def test_module_level_median_abs_dev():
    assert median_abs_dev(range(97)) == 24.0


def test_module_level_rel_std_dev():
    assert abs(rel_std_dev(range(97)) - (28.0 / 48.0)) < 1e-12


def test_module_level_skewness_kurtosis():
    assert skewness(range(97)) == 0.0
    assert kurtosis(range(9)) == 1.99125


def test_module_level_trimean_iqr():
    assert trimean(range(97)) == 48.0
    assert iqr(range(1001)) == 500


def test_module_level_pearson_type():
    # just exercise the function
    assert pearson_type(range(97)) in {0, 1, 2, 3, 4, 5, 6, 7}


# ---------------------------------------------------------------------------
# Caching / clear_cache
# ---------------------------------------------------------------------------

def test_cache_set_on_access():
    s = Stats(range(10))
    assert not hasattr(s, '_mean')
    _ = s.mean
    assert hasattr(s, '_mean')
    assert s._mean == s.mean


def test_clear_cache():
    s = Stats(range(10))
    _ = s.mean
    _ = s.variance
    assert hasattr(s, '_mean')
    assert hasattr(s, '_variance')
    s.clear_cache()
    assert not hasattr(s, '_mean')
    assert not hasattr(s, '_variance')


def test_clear_cache_recomputes():
    s = Stats([1, 2, 3])
    assert s.mean == 2.0
    s.data.append(10)
    s.clear_cache()
    assert s.mean == 4.0


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------

def test_get_quantile_median():
    assert Stats(range(100)).get_quantile(0.5) == 49.5


def test_get_quantile_min_max():
    s = Stats(range(10))
    assert s.get_quantile(0.0) == 0
    assert s.get_quantile(1.0) == 9


def test_get_quantile_empty_returns_default():
    s = Stats([], default=42.0)
    assert s.get_quantile(0.5) == 42.0


def test_get_quantile_out_of_range():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_quantile(1.5)
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)


def test_get_quantile_accepts_string():
    # float(q) conversion
    s = Stats(range(10))
    assert s.get_quantile('0.5') == 4.5


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------

def test_get_zscore():
    s = Stats([1, 2, 3, 4, 5])
    m = s.mean
    sd = s.std_dev
    assert s.get_zscore(3) == (3 - m) / sd


def test_get_zscore_zero_stddev_equal():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0


def test_get_zscore_zero_stddev_greater():
    s = Stats([5, 5, 5])
    assert s.get_zscore(10) == float('inf')


def test_get_zscore_zero_stddev_less():
    s = Stats([5, 5, 5])
    assert s.get_zscore(1) == float('-inf')


# ---------------------------------------------------------------------------
# trim_relative
# ---------------------------------------------------------------------------

def test_trim_relative_modifies_in_place():
    s = Stats(range(10))
    s.trim_relative(0.1)
    assert s.data == [1, 2, 3, 4, 5, 6, 7, 8]


def test_trim_relative_noop_when_small():
    s = Stats(range(3))
    s.trim_relative(0.1)
    assert s.data == [0, 1, 2]


def test_trim_relative_invalid():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.trim_relative(0.5)
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def test_get_histogram_counts_empty():
    s = Stats([])
    assert s.get_histogram_counts() == [(0.0, 0)]


def test_get_histogram_counts_with_bin_count():
    s = Stats(range(10))
    result = s.get_histogram_counts(bins=2)
    assert len(result) == 2
    # all counts sum to number of data points
    assert sum(c for _, c in result) == 10


def test_get_histogram_counts_with_bin_list():
    s = Stats(range(10))
    result = s.get_histogram_counts(bins=[0.0, 5.0, 9.0])
    assert sum(c for _, c in result) == 10


def test_get_histogram_counts_adds_min():
    s = Stats([2, 3, 4])
    # min is 2, passing bins starting at 3 should prepend min
    result = s.get_histogram_counts(bins=[3.0, 4.0])
    assert result[0][0] == 2.0


def test_get_histogram_counts_unexpected_kw():
    s = Stats(range(10))
    with pytest.raises(TypeError):
        s.get_histogram_counts(foo=1)


def test_format_histogram():
    data = list(range(20)) + list(range(5, 15)) + [10]
    out = Stats(data).format_histogram(width=30)
    assert isinstance(out, str)
    # five lines expected from the docstring example
    lines = out.splitlines()
    assert len(lines) == 5


def test_format_histogram_counts_basic():
    bin_counts = [(0.0, 5), (5.0, 10), (10.0, 3)]
    out = format_histogram_counts(bin_counts, width=40)
    lines = out.splitlines()
    assert len(lines) == 3
    # Each line should contain colon and a count
    assert '0.0:' in lines[0]
    assert '5.0:' in lines[1]
    assert '10.0:' in lines[2]


def test_format_histogram_counts_zero_count_uses_pipe():
    bin_counts = [(0.0, 0), (1.0, 5)]
    out = format_histogram_counts(bin_counts, width=40)
    lines = out.splitlines()
    # zero count line should contain '|'
    assert '|' in lines[0]


def test_format_histogram_counts_format_bin():
    bin_counts = [(0.0, 1), (1.0, 2)]
    out = format_histogram_counts(bin_counts, width=40,
                                  format_bin=lambda v: '%.1fms' % v)
    lines = out.splitlines()
    assert '0.0ms:' in lines[0]
    assert '1.0ms:' in lines[1]


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_dict_default():
    s = Stats(range(1, 8))
    result = s.describe()
    assert result['count'] == 7
    assert result['mean'] == 4.0
    assert result['std_dev'] == 2.0
    assert result['mad'] == 2.0
    assert result['min'] == 1
    assert result['max'] == 7
    assert result['0.25'] == 2.5
    assert result['0.5'] == 4
    assert result['0.75'] == 5.5


def test_describe_list():
    s = Stats(range(1, 8))
    result = s.describe(format='list')
    assert isinstance(result, list)
    keys = [k for k, _ in result]
    assert keys[0] == 'count'
    assert keys[-1] == 'max'
    assert '0.25' in keys
    assert '0.5' in keys
    assert '0.75' in keys


def test_describe_text():
    s = Stats(range(1, 8))
    out = s.describe(format='text')
    expected = (
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
    assert out == expected


def test_describe_invalid_format():
    s = Stats(range(1, 8))
    with pytest.raises(ValueError):
        s.describe(format='bogus')


def test_describe_custom_quantiles():
    s = Stats(range(1, 8))
    result = s.describe(quantiles=[0.1, 0.9])
    assert '0.1' in result
    assert '0.9' in result
    assert '0.5' not in result


def test_module_level_describe_text():
    out = describe(range(7), format='text')
    expected = (
        "count:    7\n"
        "mean:     3.0\n"
        "std_dev:  2.0\n"
        "mad:      2.0\n"
        "min:      0\n"
        "0.25:     1.5\n"
        "0.5:      3\n"
        "0.75:     4.5\n"
        "max:      6"
    )
    assert out == expected


def test_module_level_describe_dict():
    result = describe(range(7))
    assert result['count'] == 7
    assert result['mean'] == 3.0


# ---------------------------------------------------------------------------
# pearson_type
# ---------------------------------------------------------------------------

def test_pearson_type_symmetric():
    # symmetric distribution around median should be type 0 (normal-ish)
    # range(97) is symmetric; check it returns int in valid range
    s = Stats(range(97))
    assert s.pearson_type in {0, 1, 2, 3, 4, 5, 6, 7}


def test_pearson_type_precision_attr():
    s = Stats(range(97))
    assert s._pearson_precision == 0


# ---------------------------------------------------------------------------
# Edge cases with empty / single data
# ---------------------------------------------------------------------------

def test_empty_data_properties_return_default():
    s = Stats([], default=7.0)
    assert s.mean == 7.0
    assert s.median == 7.0
    assert s.variance == 7.0
    assert s.std_dev == 7.0
    # max/min via no data -> default
    assert s.max == 7.0
    assert s.min == 7.0


def test_empty_data_get_histogram_counts():
    s = Stats([])
    assert s.get_histogram_counts() == [(0.0, 0)]


def test_single_value_stats():
    s = Stats([42])
    assert s.mean == 42.0
    assert s.median == 42
    assert s.min == 42
    assert s.max == 42
    assert s.std_dev == 0.0
    assert s.variance == 0.0
    assert s.skewness == 0.0
    assert s.kurtosis == 0.0


def test_stats_sorted_flag_min_max():
    s = Stats([5, 4, 3, 2, 1], is_sorted=True)
    # with is_sorted, min/max use index positions
    assert s.min == 5
    assert s.max == 1
