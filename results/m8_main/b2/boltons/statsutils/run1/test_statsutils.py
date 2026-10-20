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
)


# ---------------------------------------------------------------------------
# Stats basics
# ---------------------------------------------------------------------------

def test_stats_basic_attrs():
    s = Stats(range(42))
    assert len(s) == 42
    assert list(s) == list(range(42))
    assert s.count == 42
    assert s.mean == 20.5
    assert s.min == 0
    assert s.max == 41
    assert s.median == 20.5


def test_stats_copies_data_by_default():
    data = [3, 1, 2]
    s = Stats(data)
    assert s.data is not data
    assert s.data == [3, 1, 2]
    # sorting lazily happens on the copy
    assert s.median == 2
    assert s.data == [1, 2, 3]
    assert data == [3, 1, 2]


def test_stats_no_copy():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    assert s.data is data
    assert s.median == 2
    # original not sorted / mutated
    assert data == [3, 1, 2]


def test_stats_is_sorted():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.min == 1
    assert s.max == 3
    assert s.median == 2


def test_stats_default_for_empty():
    s = Stats([], default=float('nan'))
    assert math.isnan(s.mean)
    assert math.isnan(s.median)
    assert math.isnan(s.count)
    assert math.isnan(s.max)


def test_stats_default_zero():
    s = Stats([])
    assert s.mean == 0.0
    assert s.median == 0.0
    assert s.std_dev == 0.0


def test_stats_mode():
    assert Stats([2, 1, 3, 1]).mode == 1
    assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'
    assert Stats([1, 1, 2, 2, 3]).mode == 1  # first-most-common


def test_stats_variance_std_dev():
    s = Stats(range(97))
    assert s.variance == 784.0
    assert s.std_dev == 28.0


def test_stats_iqr_trimean():
    assert Stats([1, 2, 3, 4, 5]).iqr == 2
    assert Stats(range(1001)).iqr == 500
    assert Stats([2, 1, 3]).trimean == 2.0
    assert Stats(range(97)).trimean == 48.0


def test_stats_median_abs_dev():
    s = Stats(range(97))
    assert s.median_abs_dev == 24.0
    assert s.mad == 24.0


def test_stats_rel_std_dev():
    s = Stats(range(97))
    assert round(s.rel_std_dev, 3) == 0.583


def test_stats_rel_std_dev_zero_mean():
    s = Stats([-1, 1], default=42.0)
    # mean is 0 -> default returned
    assert s.rel_std_dev == 42.0


def test_stats_skewness_symmetric():
    assert Stats(range(97)).skewness == 0.0


def test_stats_skewness_single_value_default():
    s = Stats([5], default=-7.0)
    assert s.skewness == -7.0


def test_stats_skewness_constant_default():
    s = Stats([5, 5, 5], default=-7.0)
    assert s.skewness == -7.0


def test_stats_kurtosis():
    assert Stats(range(9)).kurtosis == 1.99125


def test_stats_kurtosis_const_default():
    # returns hardcoded 0.0 (not default) for degenerate data
    s = Stats([5, 5, 5], default=99.0)
    assert s.kurtosis == 0.0


def test_stats_pearson_type_normal():
    # symmetric with beta2 == 3-ish
    s = Stats(range(9))
    # Just assert it returns one of valid types (branch executed)
    assert s.pearson_type in (0, 1, 2, 3, 4, 5, 6, 7)


def _make_pearson_fake(skewness, kurtosis, precision=0):
    # The module deletes the _calc_* helpers after wiring up module-level
    # helper functions, so use the descriptor's stored func instead.
    f = Stats.__dict__['pearson_type'].func.__get__(None)  # placeholder
    return f


def _pearson_type(skewness, kurtosis):
    # The descriptor retains the original func. Access via the class
    # attribute (the _StatsProperty descriptor).
    prop = Stats.__dict__['pearson_type']
    func = prop.func

    class FakeStats:
        _pearson_precision = 0

    f = FakeStats()
    f.skewness = skewness
    f.kurtosis = kurtosis
    return func(f)


def test_pearson_type_symmetric_beta_small():
    # c1 == 0, beta2 < 3 -> 2
    assert _pearson_type(0.0, 2.0) == 2


def test_pearson_type_symmetric_beta_large():
    # c1 == 0, beta2 > 3 -> 7
    assert _pearson_type(0.0, 4.0) == 7


def test_pearson_type_gamma():
    # b1 = 1, b2 = 4.5 => c2 = 2*4.5 - 3*1 - 6 = 0 -> 3
    assert _pearson_type(1.0, 4.5) == 3


def test_pearson_type_beta_negative_k():
    assert _pearson_type(0.5, 5.0) in (1, 4, 5, 6)


# ---------------------------------------------------------------------------
# Quantiles and zscores
# ---------------------------------------------------------------------------

def test_get_quantile():
    assert Stats(range(100)).get_quantile(0.5) == 49.5
    s = Stats([1, 2, 3, 4, 5])
    assert s.get_quantile(0.0) == 1
    assert s.get_quantile(1.0) == 5


def test_get_quantile_invalid():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.get_quantile(1.5)
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)


def test_get_quantile_empty():
    s = Stats([], default=13.0)
    assert s.get_quantile(0.5) == 13.0


def test_get_zscore():
    s = Stats([1, 2, 3, 4, 5])
    mean_ = s.mean
    sd = s.std_dev
    assert s.get_zscore(3) == pytest.approx((3 - mean_) / sd)


def test_get_zscore_zero_std():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0
    assert s.get_zscore(6) == float('inf')
    assert s.get_zscore(4) == float('-inf')


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------

def test_trim_relative():
    s = Stats(range(10))
    s.trim_relative(0.2)
    # trims 2 from each side
    assert s.data == [2, 3, 4, 5, 6, 7]


def test_trim_relative_zero_diff():
    s = Stats([1, 2, 3])
    before = list(s.data)
    s.trim_relative(0.1)  # int(3*0.1)=0 -> no change
    assert s.data == before


def test_trim_relative_invalid():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.trim_relative(0.5)
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)


# ---------------------------------------------------------------------------
# Caching / clear_cache
# ---------------------------------------------------------------------------

def test_cache_and_clear():
    s = Stats([1, 2, 3])
    assert s.mean == 2.0
    assert hasattr(s, '_mean')
    s.clear_cache()
    assert not hasattr(s, '_mean')
    # still computes correctly
    assert s.mean == 2.0


# ---------------------------------------------------------------------------
# Histogram bins
# ---------------------------------------------------------------------------

def test_get_bin_bounds_empty():
    s = Stats([])
    assert s._get_bin_bounds() == [0.0]


def test_get_bin_bounds_small_count():
    # len < 4 branch: count defaults to len_data, dx = (max-min)/count
    s = Stats([0, 1, 2])
    bins = s._get_bin_bounds()
    assert bins == [0.0, pytest.approx(2.0 / 3.0), pytest.approx(4.0 / 3.0)]


def test_get_bin_bounds_small_explicit_count():
    s = Stats([0, 10])
    bins = s._get_bin_bounds(count=2)
    assert bins == [0.0, 5.0]


def test_get_bin_bounds_with_max():
    s = Stats([0, 1, 2])
    bins = s._get_bin_bounds(with_max=True)
    assert bins[-1] == 2.0


def test_get_bin_bounds_freedman_zero_dx():
    s = Stats([5, 5, 5, 5])
    assert s._get_bin_bounds() == [5.0]
    assert s._get_bin_bounds(with_max=True) == [5.0, 5.0]


def test_get_bin_bounds_freedman_normal():
    s = Stats(list(range(20)) + list(range(5, 15)) + [10])
    bins = s._get_bin_bounds()
    assert len(bins) >= 1
    for b in bins:
        assert b < 20


def test_get_bin_bounds_explicit_count():
    s = Stats(range(20))
    bins = s._get_bin_bounds(count=4)
    assert len(bins) == 4
    assert bins[0] == 0


# ---------------------------------------------------------------------------
# get_histogram_counts
# ---------------------------------------------------------------------------

def test_histogram_counts_default():
    s = Stats(list(range(20)) + list(range(5, 15)) + [10])
    counts = s.get_histogram_counts()
    assert isinstance(counts, list)
    total = sum(c for _, c in counts)
    assert total == len(s.data)


def test_histogram_counts_int_bins():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=4)
    assert sum(c for _, c in counts) == 10


def test_histogram_counts_list_bins():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=[0.0, 5.0, 10.0])
    assert sum(c for _, c in counts) == 10
    # boundaries rounded down to 1 digit
    assert all(isinstance(b, float) for b, _ in counts)


def test_histogram_counts_list_bins_extends_min():
    s = Stats([1, 2, 3, 4, 5])
    counts = s.get_histogram_counts(bins=[3.0, 5.0])
    # min (1) prepended since 1 < 3.0
    assert counts[0][0] == 1.0
    assert sum(c for _, c in counts) == 5


def test_histogram_counts_invalid_bins():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins=[object()])


def test_histogram_counts_bin_digits():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=4, bin_digits=0)
    assert all(b == int(b) for b, _ in counts)


def test_histogram_counts_unexpected_kw():
    s = Stats([1, 2, 3])
    with pytest.raises(TypeError):
        s.get_histogram_counts(bogus=1)


# ---------------------------------------------------------------------------
# format_histogram
# ---------------------------------------------------------------------------

def test_format_histogram_doc_example():
    data = list(range(20)) + list(range(5, 15)) + [10]
    out = Stats(data).format_histogram(width=30)
    expected = (
        " 0.0:  5 #########\n"
        " 4.4:  8 ###############\n"
        " 8.9: 11 ####################\n"
        "13.3:  5 #########\n"
        "17.8:  2 ####"
    )
    assert out == expected


def test_format_histogram_format_bin():
    s = Stats(range(10))
    out = s.format_histogram(bins=2, format_bin=lambda b: '%sms' % b, width=40)
    assert 'ms' in out


def test_format_histogram_small_bars():
    # bar minimum is 3 columns
    out = format_histogram_counts([(0.0, 1)], width=1)
    assert out.endswith('|') or out.endswith('#')


# ---------------------------------------------------------------------------
# format_histogram_counts directly
# ---------------------------------------------------------------------------

def test_format_histogram_counts_basic():
    bin_counts = [(0.0, 5), (1.0, 10), (2.0, 2)]
    out = format_histogram_counts(bin_counts, width=40)
    lines = out.split('\n')
    assert len(lines) == 3
    assert lines[0].startswith('0.0:')
    assert lines[1].startswith('1.0:')


def test_format_histogram_counts_zero_count():
    # zero count -> '|' bar
    out = format_histogram_counts([(0.0, 5), (1.0, 0)], width=40)
    second = out.split('\n')[1]
    assert second.endswith('|')


def test_format_histogram_counts_format_bin():
    out = format_histogram_counts([(1.0, 3)], width=40,
                                  format_bin=lambda v: '<%s>' % v)
    assert '<1.0>' in out


def test_format_histogram_counts_default_width():
    out = format_histogram_counts([(0.0, 1), (1.0, 2)])
    assert '\n' in out


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_text_example():
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


def test_describe_dict():
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
    assert labels[0] == 'count'
    assert labels[-1] == 'max'


def test_describe_custom_quantiles():
    d = Stats(range(10)).describe(quantiles=[0.1, 0.9])
    assert '0.1' in d
    assert '0.9' in d
    assert '0.25' not in d


def test_describe_invalid_format():
    with pytest.raises(ValueError):
        Stats(range(5)).describe(format='bogus')


def test_describe_convenience():
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
    assert describe(range(7), format='text') == expected


# ---------------------------------------------------------------------------
# module-level helper functions
# ---------------------------------------------------------------------------

def test_module_mean():
    assert mean(range(42)) == 20.5


def test_module_median():
    assert median([2, 1, 3]) == 2
    assert median(range(97)) == 48


def test_module_mode():
    assert mode([2, 1, 3, 1]) == 1


def test_module_variance_std_dev():
    assert variance(range(97)) == 784.0
    assert std_dev(range(97)) == 28.0


def test_module_median_abs_dev():
    assert median_abs_dev(range(97)) == 24.0


def test_module_rel_std_dev():
    assert round(rel_std_dev(range(97)), 3) == 0.583


def test_module_skewness_kurtosis():
    assert skewness(range(97)) == 0.0
    assert kurtosis(range(9)) == 1.99125


def test_module_trimean_iqr():
    assert trimean([2, 1, 3]) == 2.0
    assert iqr([1, 2, 3, 4, 5]) == 2


def test_module_helpers_respect_default():
    assert math.isnan(mean([], default=float('nan')))
    assert median([], default=5.0) == 5.0
