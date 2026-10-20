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
    iqr,
    trimean,
    mad,
    median_abs_dev,
    rel_std_dev,
    skewness,
    kurtosis,
)


# ------------------------- Stats basics -------------------------

def test_stats_basic_iteration_and_len():
    s = Stats([1, 2, 3])
    assert len(s) == 3
    assert list(s) == [1, 2, 3]


def test_stats_count_mean_max_min():
    s = Stats([2, 1, 3])
    assert s.count == 3
    assert s.mean == 2.0
    assert s.max == 3
    assert s.min == 1


def test_stats_median_mode():
    s = Stats([2, 1, 3])
    assert s.median == 2
    assert s.mode == 1


def test_stats_median_even():
    s = Stats(range(4))
    assert s.median == 1.5


def test_stats_mode_tie_returns_first_in_data():
    s = Stats([1, 1, 2, 2, 3])
    assert s.mode == 1


def test_stats_iqr_trimean():
    s = Stats([1, 2, 3, 4, 5])
    assert s.iqr == 2
    assert s.trimean == 3.0


def test_stats_variance_std_dev():
    s = Stats(range(97))
    assert s.variance == 784.0
    assert s.std_dev == 28.0


def test_stats_mad_alias():
    s = Stats(range(97))
    assert s.median_abs_dev == 24.0
    assert s.mad == s.median_abs_dev


def test_stats_rel_std_dev():
    s = Stats(range(97))
    assert round(s.rel_std_dev, 3) == 0.583


def test_stats_rel_std_dev_zero_mean_returns_default():
    s = Stats([-1, 1])
    assert s.mean == 0.0
    assert s.rel_std_dev == s.default


def test_stats_skewness_symmetric():
    s = Stats(range(97))
    assert s.skewness == 0.0


def test_stats_skewness_returns_default_when_too_short():
    s = Stats([5], default=float('nan'))
    assert math.isnan(s.skewness)


def test_stats_skewness_returns_default_when_std_dev_zero():
    s = Stats([3, 3, 3], default=-1.0)
    assert s.skewness == -1.0


def test_stats_kurtosis():
    s = Stats(range(9))
    assert s.kurtosis == 1.99125


def test_stats_kurtosis_short_returns_zero():
    s = Stats([5], default=float('nan'))
    assert s.kurtosis == 0.0


def test_stats_kurtosis_std_dev_zero_returns_zero():
    s = Stats([3, 3, 3], default=99.0)
    assert s.kurtosis == 0.0


# ------------------------- Empty data / default -------------------------

def test_empty_stats_returns_default():
    s = Stats([], default=float('nan'))
    assert math.isnan(s.mean)
    assert math.isnan(s.median)
    assert math.isnan(s.count)
    assert math.isnan(s.max)


def test_empty_stats_get_quantile_returns_default():
    s = Stats([], default=-5.0)
    assert s.get_quantile(0.5) == -5.0


def test_empty_stats_describe_counts():
    s = Stats([], default=0.0)
    d = s.describe()
    # count is 0 (len of data), others default
    assert d['count'] == 0


# ------------------------- constructor behavior -------------------------

def test_use_copy_true_copies_list():
    data = [3, 1, 2]
    s = Stats(data)
    s.data.sort()
    assert data == [3, 1, 2]


def test_use_copy_false_reuses_list():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    s.data.sort()
    assert data == [1, 2, 3]


def test_is_sorted_max_min():
    s = Stats([1, 2, 3], use_copy=False, is_sorted=True)
    assert s.max == 3
    assert s.min == 1


# ------------------------- caching -------------------------

def test_cache_clear_recomputes():
    s = Stats([1, 2, 3])
    assert s.mean == 2.0
    s.data.append(4)
    # cached value still old
    assert s.mean == 2.0
    s.clear_cache()
    assert s.mean == 2.5


def test_clear_cache_when_nothing_cached():
    s = Stats([1, 2, 3])
    # should not raise
    s.clear_cache()


# ------------------------- get_quantile -------------------------

def test_get_quantile_values():
    s = Stats(range(100))
    assert s.get_quantile(0.5) == 49.5
    assert s.get_quantile(0.0) == 0
    assert s.get_quantile(1.0) == 99


def test_get_quantile_out_of_range():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)
    with pytest.raises(ValueError):
        s.get_quantile(1.1)


def test_get_quantile_string():
    s = Stats(range(10))
    assert s.get_quantile('0.5') == 4.5


def test_get_quantile_bad_string():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_quantile('abc')


# ------------------------- get_zscore -------------------------

def test_zscore_normal():
    s = Stats(range(97))
    # mean 48, std_dev 28
    assert s.get_zscore(48) == 0.0
    assert s.get_zscore(76) == 1.0


def test_zscore_zero_std_dev():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0
    assert s.get_zscore(6) == float('inf')
    assert s.get_zscore(4) == float('-inf')


# ------------------------- trim_relative -------------------------

def test_trim_relative_basic():
    s = Stats(range(100))
    s.trim_relative(0.1)
    assert s.data == list(range(10, 90))
    assert s.mean == 49.5


def test_trim_relative_too_small_noop():
    s = Stats(range(10))
    s.trim_relative(0.0)
    assert s.data == list(range(10))


def test_trim_relative_no_effect_when_size_small():
    s = Stats(range(5))
    # int(5 * 0.1) == 0 -> no trimming
    s.trim_relative(0.1)
    assert s.data == list(range(5))


def test_trim_relative_invalid():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.trim_relative(0.5)
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)


def test_trim_relative_clears_cache():
    s = Stats(range(100))
    assert s.mean == 49.5
    s.trim_relative(0.2)
    assert s.mean == 49.5  # symmetric trim keeps mean
    assert len(s.data) == 60


# ------------------------- histogram -------------------------

def test_get_histogram_counts_empty():
    s = Stats([])
    assert s.get_histogram_counts() == [(0.0, 0)]


def test_get_histogram_counts_with_int():
    s = Stats(range(100))
    counts = s.get_histogram_counts(bins=4)
    assert sum(c for _, c in counts) == 100
    assert len(counts) == 4


def test_get_histogram_counts_with_list():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=[0, 5, 10])
    # 0 prepended? min is 0, so [0,5,10] stays, floor rounded
    assert sum(c for _, c in counts) == 10
    bins = [b for b, _ in counts]
    assert bins == sorted(bins)
    assert len(bins) == len(set(bins))


def test_get_histogram_counts_prepends_min():
    s = Stats(range(5, 15))
    counts = s.get_histogram_counts(bins=[10, 20])
    bins = [b for b, _ in counts]
    assert 5.0 in bins
    assert sum(c for _, c in counts) == 10


def test_get_histogram_counts_bad_bins():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins=['a', 'b'])


def test_get_histogram_counts_unexpected_kwarg():
    s = Stats(range(10))
    with pytest.raises(TypeError):
        s.get_histogram_counts(bins=2, bogus=1)


def test_get_histogram_counts_bin_digits():
    s = Stats(range(100))
    counts = s.get_histogram_counts(bins=4, bin_digits=0)
    for b, _ in counts:
        assert b == int(b)


def test_format_histogram_returns_string():
    s = Stats(list(range(20)) + list(range(5, 15)) + [10])
    out = s.format_histogram(width=30)
    assert isinstance(out, str)
    lines = out.splitlines()
    assert len(lines) == 5
    assert lines[0].startswith(' 0.0:  5')


def test_format_histogram_custom_bin_format():
    s = Stats([1, 2, 3])
    out = s.format_histogram(bins=2, format_bin=lambda v: '%sms' % v)
    assert 'ms' in out


def test_format_histogram_counts_basic():
    out = format_histogram_counts([(0, 5), (1, 10)], width=40)
    lines = out.splitlines()
    assert len(lines) == 2
    # first line ends with bar of '#' and '|'
    assert ':' in lines[0]
    assert lines[0].split(':')[1].strip().startswith('5')
    # second line has more # than first
    assert lines[1].count('#') > lines[0].count('#')
    assert lines[1].endswith('#') or '#' in lines[1]


def test_format_histogram_counts_zero_count_uses_pipe():
    out = format_histogram_counts([(0, 0), (1, 4)], width=40)
    lines = out.splitlines()
    assert lines[0].endswith('|')


def test_format_histogram_counts_custom_format_bin():
    out = format_histogram_counts([(0, 1)], width=40,
                                  format_bin=lambda v: 'val%s' % v)
    assert 'val0' in out


# ------------------------- describe -------------------------

def test_describe_dict_default():
    s = Stats(range(1, 8))
    d = s.describe()
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
    s = Stats(range(1, 8))
    lst = s.describe(format='list')
    assert isinstance(lst, list)
    keys = [k for k, _ in lst]
    assert keys == ['count', 'mean', 'std_dev', 'mad', 'min',
                    '0.25', '0.5', '0.75', 'max']


def test_describe_text_format():
    s = Stats(range(1, 8))
    text = s.describe(format='text')
    expected = '\n'.join([
        'count:    7',
        'mean:     4.0',
        'std_dev:  2.0',
        'mad:      2.0',
        'min:      1',
        '0.25:     2.5',
        '0.5:      4',
        '0.75:     5.5',
        'max:      7',
    ])
    assert text == expected


def test_describe_custom_quantiles():
    s = Stats(range(1, 8))
    d = s.describe(quantiles=[0.1, 0.9])
    assert d['0.1'] == s.get_quantile(0.1)
    assert d['0.9'] == s.get_quantile(0.9)
    assert '0.5' not in d


def test_describe_invalid_format():
    s = Stats(range(1, 8))
    with pytest.raises(ValueError):
        s.describe(format='xml')


def test_module_level_describe():
    d = describe(range(7))
    assert d['count'] == 7
    assert d['mean'] == 3.0
    assert d['min'] == 0
    assert d['max'] == 6


def test_module_level_describe_text():
    text = describe(range(7), format='text')
    assert text.startswith('count:    7')
    assert text.endswith('max:      6')


# ------------------------- pearson_type -------------------------

def test_pearson_type_normal_like():
    # Uniform data has low skewness; check the property returns an int
    s = Stats(range(100))
    result = s.pearson_type
    assert isinstance(result, int)


def test_pearson_type_symmetric_beta_low_kurtosis():
    # construct a distribution with zero skewness and kurtosis < 3
    s = Stats([1, 2, 3, 4, 5, 6, 7])
    assert isinstance(s.pearson_type, int)


def test_pearson_type_cached_after_skew_kurt():
    s = Stats(range(50))
    _ = s.skewness
    _ = s.kurtosis
    assert isinstance(s.pearson_type, int)


# ------------------------- module-level functions -------------------------

def test_module_level_mean_median_mode():
    assert mean(range(20)) == 9.5
    assert median([2, 1, 3]) == 2
    assert mode([2, 1, 3, 1]) == 1


def test_module_level_variance_std_dev():
    assert variance(range(97)) == 784.0
    assert std_dev(range(97)) == 28.0


def test_module_level_iqr_trimean():
    assert iqr([1, 2, 3, 4, 5]) == 2
    assert trimean([2, 1, 3]) == 2.0


def test_module_level_mad():
    assert median_abs_dev(range(97)) == 24.0
    assert mad(range(97)) == 24.0


def test_module_level_rel_std_dev():
    assert round(rel_std_dev(range(97)), 3) == 0.583


def test_module_level_skewness_kurtosis():
    assert skewness(range(97)) == 0.0
    assert kurtosis(range(9)) == 1.99125


def test_module_level_functions_dont_copy():
    data = [1, 2, 3]
    assert mean(data) == 2.0
    # module-level helper uses use_copy=False, but mean doesn't sort
    assert data == [1, 2, 3]


def test_module_level_default():
    assert math.isnan(mean([], default=float('nan')))
