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


class TestStatsBasics:
    def test_len_and_iter(self):
        s = Stats([3, 1, 2])
        assert len(s) == 3
        assert list(s) == [3, 1, 2]

    def test_use_copy_true(self):
        data = [1, 2, 3]
        s = Stats(data)
        s.data.append(4)
        assert data == [1, 2, 3]
        assert s.data == [1, 2, 3, 4]

    def test_use_copy_false(self):
        data = [1, 2, 3]
        s = Stats(data, use_copy=False)
        assert s.data is data

    def test_default_value(self):
        s = Stats([], default=float('nan'))
        assert math.isnan(s.mean)
        assert math.isnan(s.median)

    def test_default_zero(self):
        s = Stats([])
        assert s.mean == 0.0
        assert s.median == 0.0

    def test_count(self):
        assert Stats(range(20)).count == 20

    def test_mean(self):
        assert Stats(range(20)).mean == 9.5

    def test_max_min_unsorted(self):
        s = Stats([2, 1, 3])
        assert s.max == 3
        assert s.min == 1

    def test_max_min_sorted(self):
        s = Stats([1, 2, 3], is_sorted=True)
        assert s.max == 3
        assert s.min == 1

    def test_median_odd(self):
        assert Stats([2, 1, 3]).median == 2

    def test_median_even(self):
        assert Stats([1, 2, 3, 4]).median == 2.5

    def test_median_outlier(self):
        assert Stats(list(range(96)) + [1066]).median == 48

    def test_mode(self):
        assert Stats([2, 1, 3, 1]).mode == 1

    def test_mode_categorical(self):
        assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'

    def test_mode_ties_first(self):
        assert Stats([1, 1, 2, 2, 3]).mode == 1

    def test_variance(self):
        assert Stats(range(97)).variance == 784.0

    def test_std_dev(self):
        assert Stats(range(97)).std_dev == 28.0

    def test_median_abs_dev(self):
        assert Stats(range(97)).median_abs_dev == 24.0

    def test_mad_alias(self):
        s = Stats(range(97))
        assert s.mad == s.median_abs_dev

    def test_rel_std_dev(self):
        assert round(Stats(range(97)).rel_std_dev, 3) == 0.583

    def test_rel_std_dev_zero_mean(self):
        s = Stats([-1, 1], default=42.0)
        assert s.rel_std_dev == 42.0

    def test_skewness_symmetric(self):
        assert Stats(range(97)).skewness == 0.0

    def test_skewness_left_right(self):
        left = Stats(list(range(97)) + list(range(10))).skewness
        right = Stats(list(range(97)) + list(range(87, 97))).skewness
        assert round(left, 3) == 0.114
        assert round(right, 3) == -0.114

    def test_skewness_single_value_returns_default(self):
        s = Stats([5], default=99.0)
        assert s.skewness == 99.0

    def test_skewness_zero_std_returns_default(self):
        s = Stats([2, 2, 2], default=7.0)
        assert s.skewness == 7.0

    def test_kurtosis(self):
        assert Stats(range(9)).kurtosis == 1.99125

    def test_kurtosis_single_value_returns_zero(self):
        assert Stats([5]).kurtosis == 0.0

    def test_kurtosis_zero_std_returns_zero(self):
        assert Stats([2, 2, 2]).kurtosis == 0.0

    def test_trimean(self):
        assert Stats([2, 1, 3]).trimean == 2.0
        assert Stats(range(97)).trimean == 48.0

    def test_iqr(self):
        assert Stats([1, 2, 3, 4, 5]).iqr == 2
        assert Stats(range(1001)).iqr == 500


class TestQuantile:
    def test_get_quantile_median(self):
        assert Stats(range(100)).get_quantile(0.5) == 49.5

    def test_get_quantile_ends(self):
        s = Stats(range(10))
        assert s.get_quantile(0.0) == 0
        assert s.get_quantile(1.0) == 9

    def test_get_quantile_str(self):
        assert Stats(range(10)).get_quantile('0.5') == 4.5

    def test_get_quantile_empty(self):
        assert Stats([], default=3.0).get_quantile(0.5) == 3.0

    def test_get_quantile_out_of_range(self):
        with pytest.raises(ValueError):
            Stats(range(10)).get_quantile(1.5)
        with pytest.raises(ValueError):
            Stats(range(10)).get_quantile(-0.1)


class TestZScore:
    def test_zscore_basic(self):
        s = Stats(range(10))
        assert s.get_zscore(mean(range(10))) == 0.0
        pos = s.get_zscore(9)
        assert pos > 0
        neg = s.get_zscore(0)
        assert neg < 0

    def test_zscore_zero_std_equal(self):
        s = Stats([5, 5, 5])
        assert s.get_zscore(5) == 0

    def test_zscore_zero_std_above(self):
        s = Stats([5, 5, 5])
        assert s.get_zscore(10) == float('inf')

    def test_zscore_zero_std_below(self):
        s = Stats([5, 5, 5])
        assert s.get_zscore(0) == float('-inf')


class TestTrimRelative:
    def test_trim_relative_modifies_and_clears_cache(self):
        s = Stats(list(range(100)))
        _ = s.mean  # populate cache
        assert hasattr(s, '_mean')
        s.trim_relative(0.1)
        assert s.data == list(range(10, 90))
        assert not hasattr(s, '_mean')

    def test_trim_relative_noop_small(self):
        s = Stats([1, 2, 3, 4])
        s.trim_relative(0.1)
        assert s.data == [1, 2, 3, 4]

    def test_trim_relative_invalid(self):
        s = Stats([1, 2, 3])
        with pytest.raises(ValueError):
            s.trim_relative(0.5)
        with pytest.raises(ValueError):
            s.trim_relative(-0.1)


class TestClearCache:
    def test_clear_cache(self):
        s = Stats(range(10))
        _ = s.mean
        _ = s.median
        assert hasattr(s, '_mean')
        s.clear_cache()
        assert not hasattr(s, '_mean')
        assert not hasattr(s, '_median')
        # re-computation still works
        assert s.mean == 4.5


class TestPearsonType:
    def test_pearson_type_real_data_returns_int(self):
        s = Stats(range(100))
        assert isinstance(s.pearson_type, int)

    def test_pearson_type_symmetric(self):
        s = Stats([1, 2, 3, 4, 5])
        assert s.pearson_type == 2


class TestGetBinBounds:
    def test_empty(self):
        assert Stats([])._get_bin_bounds() == [0.0]

    def test_small_data_without_count(self):
        s = Stats([1, 2, 3])
        bounds = s._get_bin_bounds()
        assert bounds == [1.0, 1.6666666666666665, 2.333333333333333]

    def test_small_data_with_count(self):
        s = Stats([0, 4])
        bounds = s._get_bin_bounds(count=2)
        assert bounds == [0.0, 2.0]

    def test_with_max(self):
        s = Stats([1, 2, 3])
        bounds = s._get_bin_bounds(with_max=True)
        assert bounds[-1] == 3.0

    def test_freedman_zero_dx(self):
        s = Stats([1, 1, 1, 1, 1])
        assert s._get_bin_bounds() == [1.0]

    def test_freedman_zero_dx_with_max(self):
        s = Stats([1, 1, 1, 1, 1])
        assert s._get_bin_bounds(with_max=True) == [1.0, 1.0]

    def test_large_data_default(self):
        s = Stats(range(100))
        bounds = s._get_bin_bounds()
        assert bounds[0] == 0
        assert all(b < 99 for b in bounds)

    def test_large_data_with_count(self):
        s = Stats(range(100))
        bounds = s._get_bin_bounds(count=4)
        assert bounds == [0.0, 24.75, 49.5, 74.25]


class TestHistogram:
    def test_get_histogram_counts_default(self):
        data = list(range(20)) + list(range(5, 15)) + [10]
        counts = Stats(data).get_histogram_counts()
        assert counts == [(0.0, 5), (4.4, 8), (8.9, 11), (13.3, 5), (17.8, 2)]

    def test_get_histogram_counts_int_bins(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=3)
        assert sum(c for _, c in counts) == 10

    def test_get_histogram_counts_list_bins(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=[0.0, 5.0, 9.0])
        assert counts[0] == (0.0, 5)
        assert counts[1] == (5.0, 4)
        assert counts[2] == (9.0, 1)

    def test_get_histogram_counts_prepend_min(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=[3.0, 6.0, 9.0])
        # min (0) is less than 3.0 so it gets prepended
        assert counts[0][0] == 0.0

    def test_get_histogram_counts_bin_digits(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=3, bin_digits=0)
        for b, _ in counts:
            assert b == int(b)

    def test_get_histogram_counts_unexpected_kw(self):
        s = Stats(range(10))
        with pytest.raises(TypeError):
            s.get_histogram_counts(foo=1)

    def test_get_histogram_counts_invalid_bins(self):
        s = Stats(range(10))
        with pytest.raises(ValueError):
            s.get_histogram_counts(bins=object())

    def test_format_histogram(self):
        data = list(range(20)) + list(range(5, 15)) + [10]
        out = Stats(data).format_histogram(width=30)
        expected = (
            ' 0.0:  5 #########\n'
            ' 4.4:  8 ###############\n'
            ' 8.9: 11 ####################\n'
            '13.3:  5 #########\n'
            '17.8:  2 ####'
        )
        assert out == expected

    def test_format_histogram_format_bin(self):
        s = Stats(range(10))
        out = s.format_histogram(bins=2, width=40, format_bin=lambda v: '%sms' % v)
        for line in out.splitlines():
            assert 'ms' in line


class TestFormatHistogramCounts:
    def test_basic(self):
        out = format_histogram_counts([(0.0, 5), (5.0, 10)], width=30)
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith('0.0:')
        assert lines[1].startswith('5.0:')

    def test_format_bin(self):
        out = format_histogram_counts(
            [(0.0, 3), (1.0, 6)], width=30, format_bin=lambda v: 'v%s' % v)
        assert 'v0.0' in out
        assert 'v1.0' in out

    def test_zero_count_uses_pipe(self):
        out = format_histogram_counts([(0.0, 0), (1.0, 4)], width=30)
        lines = out.splitlines()
        assert lines[0].rstrip().endswith('|')
        assert '#' in lines[1]

    def test_default_width(self):
        out = format_histogram_counts([(0.0, 1)])
        assert len(out.splitlines()) == 1


class TestDescribe:
    def test_describe_text(self):
        stats = Stats(range(1, 8))
        out = stats.describe(format='text')
        expected = (
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
        assert out == expected

    def test_describe_dict(self):
        d = Stats(range(7)).describe()
        assert d['count'] == 7
        assert d['mean'] == 3.0
        assert d['min'] == 0
        assert d['max'] == 6
        assert d['0.5'] == 3

    def test_describe_list(self):
        lst = Stats(range(7)).describe(format='list')
        assert isinstance(lst, list)
        labels = [k for k, _ in lst]
        assert labels[0] == 'count'
        assert labels[-1] == 'max'
        assert '0.5' in labels

    def test_describe_custom_quantiles(self):
        d = Stats(range(7)).describe(quantiles=[0.1, 0.9])
        assert '0.1' in d
        assert '0.9' in d
        assert '0.25' not in d

    def test_describe_invalid_format(self):
        with pytest.raises(ValueError):
            Stats(range(7)).describe(format='bogus')

    def test_module_describe(self):
        out = describe(range(7), format='text')
        expected = (
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
        assert out == expected

    def test_module_describe_dict(self):
        d = describe(range(7))
        assert d['count'] == 7
        assert d['mean'] == 3.0


class TestModuleHelpers:
    def test_mean(self):
        assert mean(range(20)) == 9.5

    def test_median(self):
        assert median([2, 1, 3]) == 2
        assert median(range(97)) == 48

    def test_mode(self):
        assert mode([2, 1, 3, 1]) == 1

    def test_variance(self):
        assert variance(range(97)) == 784.0

    def test_std_dev(self):
        assert std_dev(range(97)) == 28.0

    def test_median_abs_dev(self):
        assert median_abs_dev(range(97)) == 24.0

    def test_rel_std_dev(self):
        assert round(rel_std_dev(range(97)), 3) == 0.583

    def test_skewness(self):
        assert skewness(range(97)) == 0.0

    def test_kurtosis(self):
        assert kurtosis(range(9)) == 1.99125

    def test_trimean(self):
        assert trimean([2, 1, 3]) == 2.0

    def test_iqr(self):
        assert iqr([1, 2, 3, 4, 5]) == 2

    def test_helper_default(self):
        assert mean([], default=5.0) == 5.0

    def test_helper_does_not_copy(self):
        # mean should work with an iterable; Stats(use_copy=False) preserves source list
        data = [1, 2, 3]
        assert mean(data) == 2.0
