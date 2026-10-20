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
    iqr,
    trimean,
)


# ---------------------------------------------------------------------------
# Basic Stats construction / dunder methods
# ---------------------------------------------------------------------------

def test_stats_len_iter():
    s = Stats([1, 2, 3])
    assert len(s) == 3
    assert list(s) == [1, 2, 3]


def test_stats_use_copy_default():
    data = [3, 1, 2]
    s = Stats(data)
    s._get_sorted_data()
    assert s.data == [1, 2, 3]
    # original list untouched since a copy was made
    assert data == [3, 1, 2]


def test_stats_no_copy():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    # with use_copy=False, _get_sorted_data returns a sorted copy but does not
    # mutate the original data
    sorted_data = s._get_sorted_data()
    assert sorted_data == [1, 2, 3]
    assert s.data == [3, 1, 2]
    assert data == [3, 1, 2]


def test_stats_is_sorted():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.max == 3
    assert s.min == 1


# ---------------------------------------------------------------------------
# Default value behaviour for empty data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('attr', ['count', 'mean', 'max', 'min', 'median',
                                  'mode', 'iqr', 'trimean', 'variance',
                                  'std_dev', 'median_abs_dev', 'rel_std_dev',
                                  'skewness', 'kurtosis', 'pearson_type'])
def test_empty_defaults(attr):
    s = Stats([], default=-1.0)
    assert getattr(s, attr) == -1.0


def test_empty_get_quantile():
    s = Stats([], default=99.0)
    assert s.get_quantile(0.5) == 99.0


def test_empty_describe():
    d = Stats([], default=0.0).describe()
    assert d['count'] == 0.0
    assert d['min'] == 0.0
    assert d['max'] == 0.0


# ---------------------------------------------------------------------------
# Individual properties
# ---------------------------------------------------------------------------

def test_count():
    assert Stats(range(20)).count == 20


def test_mean():
    assert Stats(range(20)).mean == 9.5
    assert mean(range(42)) == 20.5


def test_mean_with_outlier():
    assert mean(list(range(19)) + [949]) == 56.0


def test_max_min():
    s = Stats([2, 1, 3])
    assert s.max == 3
    assert s.min == 1


def test_median():
    assert median([2, 1, 3]) == 2
    assert median(range(97)) == 48
    assert median(list(range(96)) + [1066]) == 48


def test_mode():
    assert mode([2, 1, 3, 1]) == 1
    assert mode(['a', 'b', 'b', 'c', 'c', 'c']) == 'c'
    assert mode([1, 1, 2, 2, 3]) == 1


def test_iqr():
    assert iqr([1, 2, 3, 4, 5]) == 2
    assert iqr(range(1001)) == 500


def test_trimean():
    assert trimean([2, 1, 3]) == 2.0
    assert trimean(range(97)) == 48.0
    assert trimean(list(range(96)) + [1066]) == 48.0


def test_variance():
    assert variance(range(97)) == 784.0


def test_std_dev():
    assert std_dev(range(97)) == 28.0


def test_median_abs_dev():
    assert median_abs_dev(range(97)) == 24.0


def test_mad_alias():
    s = Stats(range(97))
    assert s.mad == s.median_abs_dev == 24.0


def test_rel_std_dev():
    assert round(rel_std_dev(range(97)), 3) == 0.583


def test_rel_std_dev_zero_mean():
    # mean of [-1, 1] is 0 -> default
    s = Stats([-1, 1], default=123.0)
    assert s.rel_std_dev == 123.0


def test_skewness():
    assert skewness(range(97)) == 0.0
    left = skewness(list(range(97)) + list(range(10)))
    right = skewness(list(range(97)) + list(range(87, 97)))
    assert (round(left, 3), round(right, 3)) == (0.114, -0.114)


def test_skewness_single_value_default():
    s = Stats([5], default=-2.0)
    assert s.skewness == -2.0


def test_kurtosis():
    assert kurtosis(range(9)) == 1.99125


def test_kurtosis_single_value():
    s = Stats([5], default=-2.0)
    assert s.kurtosis == 0.0


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------

def test_get_quantile():
    assert Stats(range(100)).get_quantile(0.5) == 49.5
    assert Stats(range(100)).get_quantile(0.0) == 0
    assert Stats(range(100)).get_quantile(1.0) == 99


@pytest.mark.parametrize('q', [-0.1, 1.1, 2])
def test_get_quantile_invalid(q):
    with pytest.raises(ValueError):
        Stats(range(10)).get_quantile(q)


def test_get_quantile_accepts_string():
    assert Stats(range(100)).get_quantile('0.5') == 49.5


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------

def test_get_zscore():
    s = Stats([1, 2, 3])
    # mean 2, std_dev = sqrt(variance)
    assert s.get_zscore(2) == 0
    assert s.get_zscore(3) == pytest.approx((3 - 2) / s.std_dev)


def test_get_zscore_zero_std_dev():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0
    assert s.get_zscore(6) == float('inf')
    assert s.get_zscore(4) == float('-inf')


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_caching():
    s = Stats(range(10))
    assert s.mean == 4.5
    assert '_mean' in s.__dict__
    # cached
    assert s.__dict__['_mean'] == 4.5


def test_clear_cache():
    s = Stats(range(10))
    _ = s.mean
    assert '_mean' in s.__dict__
    s.clear_cache()
    assert '_mean' not in s.__dict__


def test_clear_cache_no_cached_attrs():
    s = Stats(range(10))
    # nothing cached yet -- should not raise
    s.clear_cache()


# ---------------------------------------------------------------------------
# trim_relative
# ---------------------------------------------------------------------------

def test_trim_relative():
    s = Stats(range(10))
    s.trim_relative(0.2)
    assert s.data == [2, 3, 4, 5, 6, 7]
    assert '_mean' not in s.__dict__


def test_trim_relative_zero():
    s = Stats(range(10))
    s.trim_relative(0.0)
    assert s.data == list(range(10))


@pytest.mark.parametrize('amount', [-0.1, 0.5, 0.6])
def test_trim_relative_invalid(amount):
    with pytest.raises(ValueError):
        Stats(range(10)).trim_relative(amount)


# ---------------------------------------------------------------------------
# Histogram counts
# ---------------------------------------------------------------------------

def test_get_histogram_counts_int_bins():
    s = Stats(range(10))
    hc = s.get_histogram_counts(bins=2)
    assert isinstance(hc, list)
    # bins are (bin, count) tuples
    total = sum(count for _, count in hc)
    assert total == 10


def test_get_histogram_counts_list_bins():
    s = Stats(range(10))
    hc = s.get_histogram_counts(bins=[0.0, 5.0, 10.0])
    assert sum(count for _, count in hc) == 10
    # first bin should start at min
    assert hc[0][0] == 0.0


def test_get_histogram_counts_prepends_min():
    s = Stats(range(10))
    hc = s.get_histogram_counts(bins=[3.0, 6.0, 9.0])
    # min (0) added as first bin
    assert hc[0][0] == 0.0


def test_get_histogram_counts_invalid_bins():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins=['x', 'y'])


def test_get_histogram_counts_unexpected_kw():
    s = Stats(range(10))
    with pytest.raises(TypeError):
        s.get_histogram_counts(bins=2, bogus=1)


def test_get_histogram_counts_empty():
    s = Stats([])
    assert s.get_histogram_counts() == [(0.0, 0)]


def test_get_histogram_counts_all_same():
    s = Stats([5, 5, 5, 5])
    hc = s.get_histogram_counts()
    assert sum(c for _, c in hc) == 4


# ---------------------------------------------------------------------------
# format_histogram
# ---------------------------------------------------------------------------

def test_format_histogram_matches_docstring():
    data = list(range(20)) + list(range(5, 15)) + [10]
    expected = (
        " 0.0:  5 #########\n"
        " 4.4:  8 ###############\n"
        " 8.9: 11 ####################\n"
        "13.3:  5 #########\n"
        "17.8:  2 ####"
    )
    assert Stats(data).format_histogram(width=30) == expected


# ---------------------------------------------------------------------------
# format_histogram_counts
# ---------------------------------------------------------------------------

def test_format_histogram_counts_basic():
    out = format_histogram_counts([(0.0, 5), (5.0, 10)], width=30)
    lines = out.split('\n')
    assert len(lines) == 2
    assert lines[0].startswith('0.0:')
    assert lines[1].startswith('5.0:')


def test_format_histogram_counts_format_bin():
    out = format_histogram_counts([(1.0, 2)], width=40,
                                  format_bin=lambda v: '%sms' % v)
    assert '1.0ms:' in out


def test_format_histogram_counts_zero_bin_shows_pipe():
    out = format_histogram_counts([(0.0, 0), (1.0, 5)], width=30)
    lines = out.split('\n')
    # The zero-count row should contain a pipe character for its bar
    assert '|' in lines[0]


def test_format_histogram_counts_width_minimum():
    # narrow width forces the bar_cols minimum of 3
    out = format_histogram_counts([(0.0, 1), (1.0, 10)], width=1)
    assert isinstance(out, str)
    assert len(out.split('\n')) == 2


def test_format_histogram_counts_no_width():
    # Should not raise even without width (uses terminal size or 80)
    out = format_histogram_counts([(0.0, 1), (1.0, 2)])
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_text():
    stats = Stats(range(1, 8))
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
    assert stats.describe(format='text') == expected


def test_describe_default_is_dict():
    d = Stats(range(7)).describe()
    assert isinstance(d, dict)
    assert d['count'] == 7
    assert d['mean'] == 3.0
    assert d['min'] == 0
    assert d['max'] == 6
    assert d['0.25'] == 1.5
    assert d['0.5'] == 3
    assert d['0.75'] == 4.5


def test_describe_list():
    items = Stats(range(7)).describe(format='list')
    assert isinstance(items, list)
    labels = [k for k, _ in items]
    assert labels == ['count', 'mean', 'std_dev', 'mad', 'min',
                      '0.25', '0.5', '0.75', 'max']


def test_describe_custom_quantiles():
    d = Stats(range(7)).describe(quantiles=[0.1, 0.9])
    assert d['0.1'] == pytest.approx(0.6)
    assert d['0.9'] == pytest.approx(5.4)
    assert '0.25' not in d


def test_describe_invalid_format():
    with pytest.raises(ValueError):
        Stats(range(7)).describe(format='xml')


def test_module_describe():
    text = describe(range(7), format='text')
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
    assert text == expected


def test_module_describe_dict():
    d = describe(range(7))
    assert d['count'] == 7
