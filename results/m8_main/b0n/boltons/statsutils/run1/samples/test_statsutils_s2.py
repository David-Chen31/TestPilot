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
    median_abs_dev,
    skewness,
    kurtosis,
    rel_std_dev,
    pearson_type,
)


class TestStatsBasics:
    def test_construction_copies_by_default(self):
        data = [1, 2, 3]
        s = Stats(data)
        assert s.data == [1, 2, 3]
        assert s.data is not data

    def test_construction_no_copy(self):
        data = [1, 2, 3]
        s = Stats(data, use_copy=False)
        assert s.data is data

    def test_len_and_iter(self):
        s = Stats([1, 2, 3])
        assert len(s) == 3
        assert list(s) == [1, 2, 3]

    def test_default(self):
        s = Stats([], default=float('nan'))
        assert math.isnan(s.mean)

    def test_empty_stats_returns_default(self):
        s = Stats([])
        assert s.mean == 0.0
        assert s.median == 0.0
        assert s.std_dev == 0.0

    def test_count(self):
        assert Stats(range(20)).count == 20

    def test_mean(self):
        assert Stats(range(20)).mean == 9.5

    def test_min_max(self):
        s = Stats([2, 1, 3])
        assert s.min == 1
        assert s.max == 3

    def test_min_max_sorted(self):
        s = Stats([1, 2, 3], is_sorted=True)
        assert s.min == 1
        assert s.max == 3

    def test_median(self):
        assert Stats([2, 1, 3]).median == 2
        assert Stats(range(97)).median == 48

    def test_mode(self):
        assert Stats([2, 1, 3, 1]).mode == 1
        assert Stats(['a', 'b', 'b', 'c', 'c', 'c']).mode == 'c'

    def test_mode_tie_first(self):
        assert Stats([1, 1, 2, 2, 3]).mode == 1

    def test_iqr(self):
        assert Stats([1, 2, 3, 4, 5]).iqr == 2
        assert Stats(range(1001)).iqr == 500

    def test_trimean(self):
        assert Stats([2, 1, 3]).trimean == 2.0
        assert Stats(range(97)).trimean == 48.0

    def test_variance(self):
        assert Stats(range(97)).variance == 784.0

    def test_std_dev(self):
        assert Stats(range(97)).std_dev == 28.0

    def test_median_abs_dev(self):
        assert Stats(range(97)).median_abs_dev == 24.0
        # mad is an alias
        assert Stats(range(97)).mad == 24.0

    def test_rel_std_dev(self):
        assert round(Stats(range(97)).rel_std_dev, 3) == 0.583

    def test_rel_std_dev_zero_mean(self):
        s = Stats([-1, 1], default=99.0)
        assert s.rel_std_dev == 99.0

    def test_skewness(self):
        assert Stats(range(97)).skewness == 0.0

    def test_skewness_single(self):
        s = Stats([5], default=42.0)
        assert s.skewness == 42.0

    def test_kurtosis(self):
        assert Stats(range(9)).kurtosis == 1.99125

    def test_kurtosis_single(self):
        assert Stats([5]).kurtosis == 0.0

    def test_pearson_type_normal(self):
        # symmetric data with kurtosis approx 3
        s = Stats([-1.0, 1.0, -1.0, 1.0])
        # just ensure it returns an int
        assert isinstance(s.pearson_type, int)


class TestCacheAndClear:
    def test_cache_used(self):
        s = Stats([1, 2, 3])
        m1 = s.mean
        m2 = s.mean
        assert m1 == m2
        assert hasattr(s, '_mean')

    def test_clear_cache(self):
        s = Stats([1, 2, 3])
        _ = s.mean
        assert hasattr(s, '_mean')
        s.clear_cache()
        assert not hasattr(s, '_mean')

    def test_clear_cache_no_cached(self):
        s = Stats([1, 2, 3])
        # Should not raise
        s.clear_cache()

    def test_clear_cache_with_none(self):
        s = Stats([1, 2, 3])
        _ = s.mean
        _ = s.count
        s.clear_cache()
        assert not hasattr(s, '_mean')
        assert not hasattr(s, '_count')


class TestQuantile:
    def test_get_quantile_median(self):
        assert Stats(range(100)).get_quantile(0.5) == 49.5

    def test_get_quantile_min_max(self):
        s = Stats([1, 5, 10])
        assert s.get_quantile(0.0) == 1
        assert s.get_quantile(1.0) == 10

    def test_get_quantile_interpolation(self):
        s = Stats([0, 10])
        assert s.get_quantile(0.25) == 2.5
        assert s.get_quantile(0.75) == 7.5

    def test_get_quantile_invalid(self):
        s = Stats([1, 2, 3])
        with pytest.raises(ValueError):
            s.get_quantile(-0.1)
        with pytest.raises(ValueError):
            s.get_quantile(1.1)

    def test_get_quantile_empty(self):
        s = Stats([], default=7.0)
        assert s.get_quantile(0.5) == 7.0


class TestZscore:
    def test_zscore(self):
        s = Stats(range(10))
        # mean 4.5, std_dev sqrt(variance of 0..9)
        m = s.mean
        sd = s.std_dev
        assert s.get_zscore(5) == pytest.approx((5 - m) / sd)

    def test_zscore_zero_std(self):
        s = Stats([5, 5, 5])
        assert s.get_zscore(5) == 0
        assert s.get_zscore(6) == float('inf')
        assert s.get_zscore(4) == float('-inf')


class TestTrim:
    def test_trim_relative(self):
        s = Stats(range(100))
        s.trim_relative(0.1)
        assert s.data == list(range(10, 90))

    def test_trim_relative_noop(self):
        s = Stats([1, 2, 3])
        s.trim_relative(0.0)
        assert s.data == [1, 2, 3]

    def test_trim_relative_small(self):
        s = Stats([1, 2, 3])
        # int(3 * 0.1) == 0, no trimming
        s.trim_relative(0.1)
        assert s.data == [1, 2, 3]

    def test_trim_relative_invalid(self):
        s = Stats([1, 2, 3])
        with pytest.raises(ValueError):
            s.trim_relative(-0.1)
        with pytest.raises(ValueError):
            s.trim_relative(0.5)

    def test_trim_clears_cache(self):
        s = Stats(range(100))
        _ = s.mean
        s.trim_relative(0.1)
        assert not hasattr(s, '_mean')


class TestHistogram:
    def test_histogram_counts_integer_bins(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=5)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in counts)
        assert sum(c for _, c in counts) == 10

    def test_histogram_counts_list_bins(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=[0, 5, 10])
        assert sum(c for _, c in counts) == 10

    def test_histogram_counts_adds_min(self):
        s = Stats([1, 2, 3, 4, 5])
        counts = s.get_histogram_counts(bins=[3, 4, 5])
        # min 1 < 3, so 1 should be prepended
        assert counts[0][0] == 1.0

    def test_histogram_counts_empty(self):
        s = Stats([])
        counts = s.get_histogram_counts()
        assert counts == [(0.0, 0)]

    def test_histogram_counts_bad_kwarg(self):
        s = Stats(range(10))
        with pytest.raises(TypeError):
            s.get_histogram_counts(bad_kwarg=1)

    def test_histogram_counts_invalid_bins(self):
        s = Stats(range(10))
        with pytest.raises(ValueError):
            s.get_histogram_counts(bins=object())

    def test_histogram_bin_digits(self):
        s = Stats(range(10))
        counts = s.get_histogram_counts(bins=3, bin_digits=2)
        for b, _ in counts:
            # rounded down to 2 digits
            assert round(b, 2) == b

    def test_format_histogram(self):
        s = Stats([1, 2, 3, 4, 5])
        out = s.format_histogram(bins=2, width=30)
        assert isinstance(out, str)
        assert '#' in out or '|' in out

    def test_format_histogram_wide(self):
        data = list(range(20)) + list(range(5, 15)) + [10]
        out = Stats(data).format_histogram(width=30)
        lines = out.splitlines()
        assert len(lines) > 0
        assert all(':' in line for line in lines)


class TestFormatHistogramCounts:
    def test_basic(self):
        out = format_histogram_counts([(0.0, 5), (1.0, 10)], width=40)
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith('0.0:')
        assert lines[1].startswith('1.0:')

    def test_format_bin(self):
        out = format_histogram_counts(
            [(0.0, 3)], width=40, format_bin=lambda v: '%dms' % v
        )
        assert '0ms:' in out

    def test_width_default(self):
        out = format_histogram_counts([(0.0, 1)])
        assert isinstance(out, str)

    def test_min_bar(self):
        out = format_histogram_counts([(0.0, 100), (1.0, 1)], width=20)
        lines = out.splitlines()
        # smallest non-zero count still displays '|'
        assert '|' in out or '#' in out


class TestDescribe:
    def test_describe_dict(self):
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

    def test_describe_list(self):
        s = Stats(range(1, 8))
        lst = s.describe(format='list')
        assert isinstance(lst, list)
        keys = [k for k, _ in lst]
        assert keys[0] == 'count'
        assert keys[-1] == 'max'
        assert '0.25' in keys

    def test_describe_text(self):
        s = Stats(range(1, 8))
        text = s.describe(format='text')
        assert text.startswith('count:')
        assert 'max:' in text

    def test_describe_invalid_format(self):
        s = Stats(range(1, 8))
        with pytest.raises(ValueError):
            s.describe(format='bogus')

    def test_describe_custom_quantiles(self):
        s = Stats(range(1, 8))
        d = s.describe(quantiles=[0.5])
        assert d['0.5'] == 4
        assert '0.25' not in d

    def test_module_describe(self):
        text = describe(range(7), format='text')
        assert 'count:' in text
        assert '3.0' in text

    def test_module_describe_dict(self):
        d = describe(range(7))
        assert d['count'] == 7
        assert d['mean'] == 3.0


class TestModuleFunctions:
    def test_mean(self):
        assert mean(range(20)) == 9.5

    def test_mean_outlier(self):
        assert mean(list(range(19)) + [949]) == 56.0

    def test_median(self):
        assert median([2, 1, 3]) == 2
        assert median(range(97)) == 48

    def test_mode(self):
        assert mode([2, 1, 3, 1]) == 1

    def test_variance(self):
        assert variance(range(97)) == 784.0

    def test_std_dev(self):
        assert std_dev(range(97)) == 28.0

    def test_iqr(self):
        assert iqr(range(1001)) == 500

    def test_trimean(self):
        assert trimean(range(97)) == 48.0

    def test_median_abs_dev(self):
        assert median_abs_dev(range(97)) == 24.0

    def test_skewness(self):
        assert skewness(range(97)) == 0.0

    def test_kurtosis(self):
        assert kurtosis(range(9)) == 1.99125

    def test_rel_std_dev(self):
        assert round(rel_std_dev(range(97)), 3) == 0.583

    def test_module_function_default(self):
        assert mean([], default=5.0) == 5.0

    def test_pearson_type_function(self):
        assert isinstance(pearson_type([1, 2, 3, 4, 5]), int)


class TestBinningEdge:
    def test_few_values(self):
        s = Stats([1, 2])
        counts = s.get_histogram_counts()
        assert sum(c for _, c in counts) == 2

    def test_constant_data(self):
        s = Stats([5, 5, 5, 5, 5])
        counts = s.get_histogram_counts()
        assert sum(c for _, c in counts) == 5

    def test_with_max(self):
        s = Stats(range(100))
        bounds = s._get_bin_bounds(with_max=True)
        assert bounds[-1] == 99.0
</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</parameter>
</invoke>
