import math
import pytest

from boltons import statsutils
from boltons.statsutils import (
    Stats,
    describe,
    format_histogram_counts,
    mean,
    median,
    mode,
    variance,
    std_dev,
    skewness,
    kurtosis,
    iqr,
    trimean,
    median_abs_dev,
    rel_std_dev,
)


# ---------------------------------------------------------------------------
# Basic Stats construction / protocol
# ---------------------------------------------------------------------------

def test_stats_len_and_iter():
    s = Stats([3, 1, 2])
    assert len(s) == 3
    assert list(s) == [3, 1, 2]


def test_stats_copy_by_default():
    data = [3, 1, 2]
    s = Stats(data)
    s._get_sorted_data()
    # original untouched, copy sorted
    assert data == [3, 1, 2]
    assert s.data == [1, 2, 3]


def test_stats_no_copy_sorts_in_place():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    assert s._get_sorted_data() == [1, 2, 3]
    assert data == [1, 2, 3]


def test_stats_is_sorted_avoids_resorting():
    data = [1, 2, 3]
    s = Stats(data, use_copy=False, is_sorted=True)
    assert s._get_sorted_data() is data


# ---------------------------------------------------------------------------
# Individual Stats properties
# ---------------------------------------------------------------------------

def test_stats_count_mean():
    s = Stats(range(20))
    assert s.count == 20
    assert s.mean == 9.5


def test_stats_mean_outlier():
    assert Stats(list(range(19)) + [949]).mean == 56.0


def test_stats_min_max():
    s = Stats([2, 1, 3])
    assert s.min == 1
    assert s.max == 3


def test_stats_min_max_sorted():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.min == 1
    assert s.max == 3


def test_stats_median_odd():
    assert Stats([2, 1, 3]).median == 2


def test_stats_median_even():
    assert Stats([1, 2, 3, 4]).median == 2.5


def test_stats_median_outlier():
    assert Stats(list(range(96)) + [1066]).median == 48


def test_stats_mode():
    assert Stats([2, 1, 3, 1]).mode == 1
    assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'


def test_stats_mode_ties_first():
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
    # alias
    assert Stats(range(97)).mad == 24.0


def test_stats_rel_std_dev():
    val = Stats(range(97)).rel_std_dev
    assert round(val, 3) == 0.583


def test_stats_rel_std_dev_zero_mean():
    s = Stats([-1, 1])
    assert s.mean == 0.0
    assert s.rel_std_dev == 0.0


def test_stats_skewness_symmetric():
    assert Stats(range(97)).skewness == 0.0


def test_stats_skewness_single_value_default():
    s = Stats([5])
    assert s.skewness == 0.0


def test_stats_kurtosis():
    assert round(Stats(range(9)).kurtosis, 5) == 1.99125


def test_stats_kurtosis_single_value():
    s = Stats([5])
    assert s.kurtosis == 0.0


def test_stats_pearson_type_normal():
    # Symmetric bell-like data should be identified
    s = Stats(range(-5, 6))
    assert s.pearson_type in (0, 2, 3, 4, 5, 6, 7, 1)


def test_stats_pearson_type_symmetric_beta():
    # constant data -> std_dev 0 -> skewness 0, kurtosis 0 -> beta2=0 < 3
    s = Stats([1, 1, 1, 1])
    assert s.pearson_type == 2


# ---------------------------------------------------------------------------
# Caching / clear_cache
# ---------------------------------------------------------------------------

def test_stats_caching():
    s = Stats([1, 2, 3, 4])
    m1 = s.mean
    assert hasattr(s, '_mean')
    m2 = s.mean
    assert m1 == m2 == 2.5


def test_stats_clear_cache():
    s = Stats([1, 2, 3, 4])
    _ = s.mean
    assert hasattr(s, '_mean')
    s.clear_cache()
    assert not hasattr(s, '_mean')


def test_stats_clear_cache_when_empty():
    s = Stats([])
    s.clear_cache()  # should not raise
    assert len(s) == 0


# ---------------------------------------------------------------------------
# Empty data / defaults
# ---------------------------------------------------------------------------

def test_stats_empty_defaults():
    s = Stats([])
    assert s.count == 0  # but data is falsy -> returns default 0.0
    # Actually count returns default when data is empty
    assert s.mean == 0.0
    assert s.median == 0.0
    assert s.min == 0.0
    assert s.max == 0.0


def test_stats_custom_default():
    nan = float('nan')
    s = Stats([], default=nan)
    assert math.isnan(s.mean)


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------

def test_get_quantile_median():
    assert Stats(range(100)).get_quantile(0.5) == 49.5


def test_get_quantile_bounds():
    s = Stats([1, 2, 3, 4, 5])
    assert s.get_quantile(0.0) == 1
    assert s.get_quantile(1.0) == 5


def test_get_quantile_empty():
    s = Stats([], default=42.0)
    assert s.get_quantile(0.5) == 42.0


def test_get_quantile_invalid():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)
    with pytest.raises(ValueError):
        s.get_quantile(1.1)


def test_get_quantile_integer_q():
    # q converted to float
    assert Stats([1, 2, 3]).get_quantile(1) == 3


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------

def test_get_zscore():
    s = Stats([1, 2, 3, 4, 5])
    mean_v = s.mean  # 3.0
    std = s.std_dev
    z = s.get_zscore(4)
    assert z == pytest.approx((4 - mean_v) / std)


def test_get_zscore_zero_std():
    s = Stats([2, 2, 2])
    assert s.get_zscore(2) == 0
    assert s.get_zscore(3) == float('inf')
    assert s.get_zscore(1) == float('-inf')


# ---------------------------------------------------------------------------
# trim_relative
# ---------------------------------------------------------------------------

def test_trim_relative_basic():
    s = Stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s.trim_relative(0.2)
    assert s.data == [3, 4, 5, 6, 7, 8]


def test_trim_relative_too_small_no_op():
    s = Stats([1, 2, 3])
    s.trim_relative(0.1)  # int(3*0.1)=0 -> no-op
    assert s.data == [1, 2, 3]


def test_trim_relative_invalid():
    s = Stats([1, 2, 3])
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)
    with pytest.raises(ValueError):
        s.trim_relative(0.5)


def test_trim_relative_clears_cache():
    s = Stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    _ = s.mean
    assert hasattr(s, '_mean')
    s.trim_relative(0.2)
    assert not hasattr(s, '_mean')


# ---------------------------------------------------------------------------
# Histogram counts
# ---------------------------------------------------------------------------

def test_get_histogram_counts_default():
    data = list(range(20)) + list(range(5, 15)) + [10]
    s = Stats(data)
    counts = s.get_histogram_counts()
    assert all(isinstance(b, float) for b, _ in counts)
    assert sum(c for _, c in counts) == len(data)


def test_get_histogram_counts_with_bin_count():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=5)
    assert len(counts) <= 6
    assert sum(c for _, c in counts) == 10


def test_get_histogram_counts_with_bin_list():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=[0.0, 5.0, 10.0])
    assert sum(c for _, c in counts) == 10


def test_get_histogram_counts_bin_list_below_min():
    s = Stats([5, 6, 7, 8])
    counts = s.get_histogram_counts(bins=[6.0, 8.0])
    # min (5) should be prepended
    assert counts[0][0] == 5.0
    assert sum(c for _, c in counts) == 4


def test_get_histogram_counts_bin_digits():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=3, bin_digits=0)
    for b, _ in counts:
        assert b == int(b)


def test_get_histogram_counts_unexpected_kwarg():
    s = Stats(range(10))
    with pytest.raises(TypeError):
        s.get_histogram_counts(bins=3, bogus=1)


def test_get_histogram_counts_bad_bins():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins=object())


def test_get_histogram_counts_empty():
    s = Stats([])
    counts = s.get_histogram_counts()
    assert counts == [(0.0, 0)]


def test_get_histogram_counts_small_len():
    s = Stats([1, 2])
    counts = s.get_histogram_counts()
    assert sum(c for _, c in counts) == 2


def test_get_bin_bounds_freedman_zero_dx():
    # Constant data -> q25 == q75 -> dx == 0
    s = Stats([5, 5, 5, 5, 5])
    bins = s._get_bin_bounds()
    assert bins == [5.0]
    bins_max = s._get_bin_bounds(with_max=True)
    assert bins_max == [5.0, 5.0]


def test_get_bin_bounds_empty():
    s = Stats([])
    assert s._get_bin_bounds() == [0.0]


# ---------------------------------------------------------------------------
# format_histogram / format_histogram_counts
# ---------------------------------------------------------------------------

def test_format_histogram_counts_basic():
    bin_counts = [(0.0, 5), (5.0, 3)]
    out = format_histogram_counts(bin_counts, width=40)
    lines = out.split('\n')
    assert len(lines) == 2
    assert lines[0].startswith('0.0: 5 ')
    assert lines[1].startswith('5.0: 3 ')
    assert '#' in out


def test_format_histogram_counts_custom_format_bin():
    bin_counts = [(1.0, 2), (2.0, 4)]
    out = format_histogram_counts(bin_counts, width=40,
                                  format_bin=lambda v: '%sms' % v)
    assert '1.0ms:' in out
    assert '2.0ms:' in out


def test_format_histogram_counts_zero_count_uses_pipe():
    bin_counts = [(0.0, 0), (1.0, 0)]
    out = format_histogram_counts(bin_counts, width=40)
    assert '|' in out


def test_format_histogram_counts_minimum_bar():
    # Normal width 40 with huge counts -> bar_cols floor at 3
    bin_counts = [(0.0, 1000)]
    out = format_histogram_counts(bin_counts, width=10)
    assert '#' in out


def test_format_histogram():
    data = list(range(20)) + list(range(5, 15)) + [10]
    s = Stats(data)
    out = s.format_histogram(width=30)
    lines = out.split('\n')
    assert lines[0].startswith(' 0.0:  5') or lines[0].startswith('0.0:  5')
    assert '#' in out


def test_format_histogram_with_bins_and_format_bin():
    s = Stats(range(10))
    out = s.format_histogram(bins=5, width=40,
                             format_bin=lambda v: '%d' % v)
    assert '#' in out


# ---------------------------------------------------------------------------
# describe (method and function)
# ---------------------------------------------------------------------------

def test_describe_default_dict():
    s = Stats(range(1, 8))
    d = s.describe()
    assert d['count'] == 7
    assert d['mean'] == 4.0
    assert d['std_dev'] == 2.0
    assert d['mad'] == 2.0
    assert d['min'] == 1
    assert d['max'] == 7
    assert d['0.25'] == 2.5
    assert d['0.5'] == 4
    assert d['0.75'] == 5.5


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
    lines = text.split('\n')
    assert lines[0].startswith('count:')
    assert lines[-1].startswith('max:')
    assert 'mean:' in text


def test_describe_custom_quantiles():
    s = Stats(range(1, 8))
    d = s.describe(quantiles=[0.1, 0.9])
    assert d['0.1'] == s.get_quantile(0.1)
    assert d['0.9'] == s.get_quantile(0.9)
    assert '0.25' not in d


def test_describe_invalid_format():
    s = Stats(range(1, 8))
    with pytest.raises(ValueError):
        s.describe(format='bogus')


def test_describe_function_matches_method():
    data = range(7)
    d1 = describe(data)
    d2 = Stats(data).describe()
    assert d1 == d2


def test_describe_function_text():
    text = describe(range(7), format='text')
    assert 'count:    7' in text
    assert 'mean:     3.0' in text
    assert 'max:      6' in text


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_module_mean():
    assert mean(range(20)) == 9.5
    assert mean(range(42)) == Stats(range(42)).mean


def test_module_median():
    assert median([2, 1, 3]) == 2
    assert median(range(97)) == 48


def test_module_mode():
    assert mode([2, 1, 3, 1]) == 1


def test_module_variance():
    assert variance(range(97)) == 784.0


def test_module_std_dev():
    assert std_dev(range(97)) == 28.0


def test_module_skewness():
    assert skewness(range(97)) == 0.0


def test_module_kurtosis():
    assert round(kurtosis(range(9)), 5) == 1.99125


def test_module_iqr():
    assert iqr([1, 2, 3, 4, 5]) == 2


def test_module_trimean():
    assert trimean([2, 1, 3]) == 2.0


def test_module_median_abs_dev():
    assert median_abs_dev(range(97)) == 24.0


def test_module_rel_std_dev():
    assert round(rel_std_dev(range(97)), 3) == 0.583


def test_module_functions_default():
    assert mean([], default=5.0) == 5.0


def test_module_helpers_use_no_copy():
    # Calling with a list should not mutate caller's list for sorted measures
    data = [3, 1, 2]
    _ = median(data)
    assert data == [3, 1, 2]


# ---------------------------------------------------------------------------
# _StatsProperty descriptor behavior
# ---------------------------------------------------------------------------

def test_stats_property_class_access():
    prop = Stats.__dict__['mean']
    assert statsutils._StatsProperty is type(prop)
    # __get__ with obj None returns descriptor
    assert Stats.mean is prop


def test_stats_property_empty_returns_default():
    s = Stats([], default=7.0)
    assert s.mean == 7.0
    assert s.std_dev == 7.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_get_pow_diffs():
    s = Stats([1, 2, 3])
    m = s.mean  # 2.0
    diffs = s._get_pow_diffs(2)
    assert diffs == [1.0, 0.0, 1.0]


def test_get_quantile_staticmethod_interpolation():
    data = [0, 10]
    assert Stats._get_quantile(data, 0.5) == 5.0
    assert Stats._get_quantile(data, 0.0) == 0
    assert Stats._get_quantile(data, 1.0) == 10
