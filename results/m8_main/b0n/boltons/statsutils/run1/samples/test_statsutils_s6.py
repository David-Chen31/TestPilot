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
    pearson_type,
)


# ---------------- Module-level function tests ----------------

def test_mean_basic():
    assert mean(range(20)) == 9.5
    assert mean([1, 2, 3]) == 2.0


def test_mean_outlier():
    assert mean(list(range(19)) + [949]) == 56.0


def test_median_odd_even():
    assert median([2, 1, 3]) == 2
    assert median(range(97)) == 48
    assert median([1, 2, 3, 4]) == 2.5


def test_median_outlier():
    assert median(list(range(96)) + [1066]) == 48


def test_mode():
    assert mode([2, 1, 3, 1]) == 1
    assert mode(['a', 'b', 'b', 'c', 'c', 'c']) == 'c'


def test_mode_tie_returns_first():
    assert mode([1, 1, 2, 2, 3]) == 1


def test_variance():
    assert variance(range(97)) == 784.0


def test_std_dev():
    assert std_dev(range(97)) == 28.0


def test_median_abs_dev():
    assert median_abs_dev(range(97)) == 24.0


def test_rel_std_dev():
    assert round(rel_std_dev(range(97)), 3) == 0.583


def test_skewness_symmetric():
    assert skewness(range(97)) == 0.0


def test_skewness_left_right():
    left_skewed = skewness(list(range(97)) + list(range(10)))
    right_skewed = skewness(list(range(97)) + list(range(87, 97)))
    assert (round(left_skewed, 3), round(right_skewed, 3)) == (0.114, -0.114)


def test_kurtosis():
    assert kurtosis(range(9)) == 1.99125


def test_iqr():
    assert iqr([1, 2, 3, 4, 5]) == 2
    assert iqr(range(1001)) == 500


def test_trimean():
    assert trimean([2, 1, 3]) == 2.0
    assert trimean(range(97)) == 48.0
    assert trimean(list(range(96)) + [1066]) == 48.0


def test_pearson_type_normal_check():
    # symmetric bell-ish data around mean
    s = Stats(range(9))
    # just check it returns a valid integer type
    assert isinstance(s.pearson_type, int)


# ---------------- Stats class basics ----------------

def test_stats_basic():
    s = Stats(range(42))
    assert s.mean == 20.5
    assert len(s) == 42
    assert list(iter(s)) == list(range(42))
    assert s.count == 42


def test_stats_min_max_unsorted():
    s = Stats([2, 1, 3])
    assert s.max == 3
    assert s.min == 1


def test_stats_min_max_sorted():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.max == 3
    assert s.min == 1


def test_stats_median():
    assert Stats([2, 1, 3]).median == 2
    assert Stats(range(97)).median == 48


def test_stats_mode():
    assert Stats([2, 1, 3, 1]).mode == 1


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
    assert s.mad == 24.0


def test_stats_skewness_kurtosis():
    s = Stats(range(97))
    assert s.skewness == 0.0
    s2 = Stats(range(9))
    assert s2.kurtosis == 1.99125


def test_stats_pearson_type():
    s = Stats(range(9))
    assert s.pearson_type in range(0, 8)


# ---------------- Empty data / default ----------------

def test_stats_empty_default():
    s = Stats([])
    assert s.default == 0.0
    assert s.count == 0
    assert s.mean == 0.0
    assert s.median == 0.0
    assert s.min == 0.0
    assert s.max == 0.0
    assert s.std_dev == 0.0


def test_stats_empty_custom_default():
    s = Stats([], default=float('nan'))
    assert math.isnan(s.mean)
    assert s.default != s.default  # nan


def test_stats_empty_quantile():
    s = Stats([], default=-1.0)
    assert s.get_quantile(0.5) == -1.0


# ---------------- get_quantile ----------------

def test_get_quantile():
    s = Stats(range(100))
    assert s.get_quantile(0.5) == 49.5
    assert s.get_quantile(0) == 0
    assert s.get_quantile(1) == 99


def test_get_quantile_out_of_range():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)
    with pytest.raises(ValueError):
        s.get_quantile(1.1)


# ---------------- get_zscore ----------------

def test_zscore():
    s = Stats(range(97))
    # mean 48, std_dev 28
    assert s.get_zscore(48) == 0.0
    assert s.get_zscore(76) == 1.0


def test_zscore_zero_std_dev():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0
    assert s.get_zscore(6) == float('inf')
    assert s.get_zscore(4) == float('-inf')


# ---------------- clear_cache ----------------

def test_clear_cache():
    s = Stats([1, 2, 3])
    _ = s.mean
    assert hasattr(s, '_mean')
    s.clear_cache()
    assert not hasattr(s, '_mean')


def test_clear_cache_empty():
    s = Stats([1, 2, 3])
    s.clear_cache()  # no cached values; should not raise


# ---------------- trimming ----------------

def test_trim_relative():
    s = Stats(range(10))
    s.trim_relative(0.2)
    assert s.data == [2, 3, 4, 5, 6, 7]


def test_trim_relative_noop():
    s = Stats(range(3))
    s.trim_relative(0.2)  # size_diff -> 0, no change
    assert s.data == [0, 1, 2]


def test_trim_relative_invalid():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.trim_relative(0.5)
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)


def test_trim_relative_caches_cleared():
    s = Stats(range(10))
    _ = s.mean
    s.trim_relative(0.2)
    assert not hasattr(s, '_mean')


# ---------------- use_copy behavior ----------------

def test_use_copy_true():
    data = [1, 2, 3]
    s = Stats(data)
    assert s.data is not data
    s.data.append(4)
    assert data == [1, 2, 3]


def test_use_copy_false():
    data = [1, 2, 3]
    s = Stats(data, use_copy=False)
    assert s.data is data


def test_use_copy_false_sorted_does_not_mutate():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    s.median
    # _get_sorted_data returns sorted copy without mutating when use_copy is False
    assert data == [3, 1, 2]


# ---------------- describe ----------------

def test_describe_dict():
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


def test_describe_list():
    s = Stats(range(1, 8))
    lst = s.describe(format='list')
    assert isinstance(lst, list)
    keys = [k for k, _ in lst]
    assert keys == ['count', 'mean', 'std_dev', 'mad', 'min',
                    '0.25', '0.5', '0.75', 'max']
    d = dict(lst)
    assert d['max'] == 7
    assert d['0.5'] == 4


def test_describe_text():
    s = Stats(range(1, 8))
    text = s.describe(format='text')
    lines = text.split('\n')
    assert len(lines) == 9
    assert lines[0].startswith('count:')
    assert lines[0].strip() == 'count:    7'
    assert lines[-1].strip() == 'max:      7'


def test_describe_custom_quantiles():
    s = Stats(range(10))
    d = s.describe(quantiles=[0.5], format='dict')
    assert d['0.5'] == 4.5
    assert '0.25' not in d


def test_describe_invalid_format():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.describe(format='json')


def test_describe_module_function():
    d = describe(range(7))
    assert d['count'] == 7
    assert d['mean'] == 3.0
    assert d['min'] == 0
    assert d['max'] == 6


# ---------------- histogram counts ----------------

def test_histogram_counts_fixed_bins():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=2)
    # two bins starting at min -> bins from 0..9
    assert isinstance(counts, list)
    assert sum(c for _, c in counts) == 10
    for b, _ in counts:
        assert isinstance(b, float)


def test_histogram_counts_list_bins():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=[0, 5, 10])
    assert sum(c for _, c in counts) == 10
    # bins sorted and include 0..5, 10 (with min added)
    vals = [b for b, _ in counts]
    assert vals == sorted(vals)


def test_histogram_counts_empty():
    s = Stats([])
    counts = s.get_histogram_counts()
    assert counts == [(0.0, 0)]


def test_histogram_counts_unexpected_kwarg():
    s = Stats(range(10))
    with pytest.raises(TypeError):
        s.get_histogram_counts(bins=2, foo=1)


def test_histogram_counts_invalid_bins():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins='not-valid-bins')


def test_histogram_counts_bin_digits():
    s = Stats(range(10))
    counts1 = s.get_histogram_counts(bins=2, bin_digits=1)
    counts0 = s.get_histogram_counts(bins=2, bin_digits=0)
    # bin_digits=0 rounds down to integer bin boundaries
    for b, _ in counts0:
        assert b == int(b)


# ---------------- format_histogram ----------------

def test_format_histogram():
    data = list(range(20)) + list(range(5, 15)) + [10]
    out = Stats(data).format_histogram(width=30)
    lines = out.split('\n')
    assert len(lines) == 5
    assert lines[0].startswith(' 0.0:')
    assert '#' in lines[0]


def test_format_histogram_counts_direct():
    bin_counts = [(0.0, 1), (1.0, 2), (2.0, 3)]
    out = format_histogram_counts(bin_counts, width=20)
    lines = out.split('\n')
    assert len(lines) == 3
    assert lines[2].split()[1] == '3'
    assert lines[2].count('#') > 0


def test_format_histogram_counts_zero_counts():
    bin_counts = [(0.0, 0), (1.0, 0)]
    out = format_histogram_counts(bin_counts, width=20)
    assert '|' in out
    lines = out.split('\n')
    assert len(lines) == 2


def test_format_histogram_counts_custom_format_bin():
    bin_counts = [(0.0, 1), (1.0, 2)]
    out = format_histogram_counts(bin_counts, width=20,
                                  format_bin=lambda v: '%dms' % int(v))
    assert '0ms' in out
    assert '1ms' in out


def test_format_histogram_counts_wide_labels_align():
    bin_counts = [(0.0, 1), (1000.0, 2)]
    out = format_histogram_counts(bin_counts, width=40)
    lines = out.split('\n')
    # label column widths should match across lines
    label_len = [len(l.split(':')[0]) for l in lines]
    assert len(set(label_len)) == 1


# ---------------- q vs Stats caching ----------------

def test_stats_caching():
    s = Stats(range(100))
    v1 = s.std_dev
    assert hasattr(s, '_std_dev')
    v2 = s.std_dev
    assert v1 == v2


def test_stats_iter():
    s = Stats([1, 2, 3])
    assert list(s) == [1, 2, 3]


def test_stats_len_matches_data():
    s = Stats([1, 2, 3, 4])
    assert len(s) == 4
    assert s.count == len(s)
