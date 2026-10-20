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
# Stats construction / basic protocol
# ---------------------------------------------------------------------------

def test_stats_len_and_iter():
    s = Stats([3, 1, 2])
    assert len(s) == 3
    assert list(s) == [3, 1, 2]


def test_stats_use_copy_default_isolates_data():
    data = [3, 1, 2]
    s = Stats(data)
    assert s.data == [3, 1, 2]
    assert s.data is not data


def test_stats_use_copy_false_shares_data():
    data = [3, 1, 2]
    s = Stats(data, use_copy=False)
    assert s.data is data


def test_stats_custom_default():
    s = Stats([1, 2, 3], default=float("nan"))
    assert math.isnan(Stats([], default=float("nan")).mean)


# ---------------------------------------------------------------------------
# Simple properties
# ---------------------------------------------------------------------------

def test_count():
    assert Stats(range(20)).count == 20
    assert Stats([]).count == 0


def test_mean():
    assert Stats(range(20)).mean == 9.5
    assert Stats([1, 2, 3]).mean == 2.0


def test_max_min():
    assert Stats([2, 1, 3]).max == 3
    assert Stats([2, 1, 3]).min == 1


def test_max_min_sorted_path():
    s = Stats([1, 2, 3], is_sorted=True)
    assert s.max == 3
    assert s.min == 1


def test_empty_defaults():
    s = Stats([])
    assert s.mean == 0.0
    assert s.max == 0.0
    assert s.min == 0.0
    assert s.count == 0.0
    assert s.mode == 0.0


def test_empty_custom_default():
    s = Stats([], default=float("nan"))
    assert math.isnan(s.mean)
    assert math.isnan(s.median)


# ---------------------------------------------------------------------------
# Central tendency
# ---------------------------------------------------------------------------

def test_median():
    assert Stats([2, 1, 3]).median == 2
    assert Stats(range(97)).median == 48


def test_mode():
    assert Stats([2, 1, 3, 1]).mode == 1
    assert Stats(["a", "b", "b", "c", "c", "c"]).mode == "c"
    # tie -> first appearing
    assert Stats([1, 1, 2, 2, 3]).mode == 1


def test_trimean():
    assert Stats([2, 1, 3]).trimean == 2.0
    assert Stats(range(97)).trimean == 48.0


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

def test_variance():
    assert Stats(range(97)).variance == 784.0


def test_std_dev():
    assert Stats(range(97)).std_dev == 28.0


def test_median_abs_dev_and_alias():
    assert Stats(range(97)).median_abs_dev == 24.0
    assert Stats(range(97)).mad == 24.0


def test_rel_std_dev():
    assert abs(Stats(range(97)).rel_std_dev - 0.583) < 1e-3


def test_rel_std_dev_zero_mean_uses_default():
    assert Stats([-1, 1], default=123.0).rel_std_dev == 123.0


def test_iqr():
    assert Stats([1, 2, 3, 4, 5]).iqr == 2
    assert Stats(range(1001)).iqr == 500


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def test_skewness():
    assert Stats(range(97)).skewness == 0.0
    left = Stats(list(range(97)) + list(range(10))).skewness
    right = Stats(list(range(97)) + list(range(87, 97))).skewness
    assert round(left, 3) == 0.114
    assert round(right, 3) == -0.114


def test_skewness_single_value_default():
    assert Stats([5]).skewness == 0.0


def test_skewness_zero_std_dev_default():
    assert Stats([5, 5, 5], default=99.0).skewness == 99.0


def test_kurtosis():
    assert Stats(range(9)).kurtosis == 1.99125


def test_kurtosis_single_value_returns_zero():
    assert Stats([5]).kurtosis == 0.0


# ---------------------------------------------------------------------------
# Pearson type
# ---------------------------------------------------------------------------

def test_pearson_type_normal():
    # symmetric data ~ type 0
    assert Stats(range(50)).pearson_type == 0


def test_pearson_type_precision_attribute():
    s = Stats(range(50))
    assert s._pearson_precision == 0


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

def test_cache_reuse_and_clear():
    s = Stats([1, 2, 3, 4])
    first_mean = s.mean
    # the cached attribute should exist now
    assert hasattr(s, "_mean")
    assert s.mean == first_mean
    s.data.append(10)
    # cached value still stale
    assert s.mean == first_mean
    s.clear_cache()
    assert s.mean == sum([1, 2, 3, 4, 10]) / 5.0


def test_clear_cache_when_nothing_cached():
    s = Stats([1, 2, 3])
    # should not raise
    s.clear_cache()


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------

def test_get_quantile():
    assert Stats(range(100)).get_quantile(0.5) == 49.5
    assert Stats(range(101)).get_quantile(0.0) == 0
    assert Stats(range(101)).get_quantile(1.0) == 100


def test_get_quantile_empty_default():
    assert Stats([], default=7.0).get_quantile(0.5) == 7.0


def test_get_quantile_bad_range():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_quantile(-0.1)
    with pytest.raises(ValueError):
        s.get_quantile(1.1)


def test_get_quantile_string_input():
    # q is coerced to float
    assert Stats(range(100)).get_quantile("0.5") == 49.5


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------

def test_get_zscore():
    s = Stats(range(10))
    m = s.mean
    sd = s.std_dev
    assert s.get_zscore(m) == pytest.approx(0.0)
    # value above mean
    assert s.get_zscore(m + sd) == pytest.approx(1.0)


def test_get_zscore_zero_std_dev():
    s = Stats([5, 5, 5])
    assert s.get_zscore(5) == 0
    assert s.get_zscore(6) == float("inf")
    assert s.get_zscore(4) == float("-inf")


# ---------------------------------------------------------------------------
# trim_relative
# ---------------------------------------------------------------------------

def test_trim_relative():
    s = Stats(range(100))
    s.trim_relative(0.1)
    assert s.data == list(range(10, 90))


def test_trim_relative_no_change_small():
    s = Stats(range(5))
    s.trim_relative(0.1)  # int(5*0.1)=0 -> no change
    assert sorted(s.data) == list(range(5))


def test_trim_relative_bad_amount():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.trim_relative(-0.1)
    with pytest.raises(ValueError):
        s.trim_relative(0.5)


# ---------------------------------------------------------------------------
# Histogram counts
# ---------------------------------------------------------------------------

def test_get_histogram_counts_basic():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=2)
    assert len(counts) == 2
    # all data accounted for
    assert sum(c for _, c in counts) == 10


def test_get_histogram_counts_explicit_bins():
    s = Stats(range(10))
    counts = s.get_histogram_counts(bins=[0.0, 5.0, 9.0])
    total = sum(c for _, c in counts)
    assert total == 10


def test_get_histogram_counts_bad_kwargs():
    s = Stats(range(10))
    with pytest.raises(TypeError):
        s.get_histogram_counts(bins=2, bogus=1)


def test_get_histogram_counts_empty():
    s = Stats([])
    counts = s.get_histogram_counts()
    assert counts == [(0.0, 0)]


def test_get_histogram_counts_invalid_bins():
    s = Stats(range(10))
    with pytest.raises(ValueError):
        s.get_histogram_counts(bins=[object(), object()])


# ---------------------------------------------------------------------------
# format_histogram
# ---------------------------------------------------------------------------

def test_format_histogram_example():
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


# ---------------------------------------------------------------------------
# describe (method + module function)
# ---------------------------------------------------------------------------

def test_describe_dict():
    s = Stats(range(1, 8))
    d = s.describe()
    assert d["count"] == 7
    assert d["mean"] == 4.0
    assert d["std_dev"] == 2.0
    assert d["mad"] == 2.0
    assert d["min"] == 1
    assert d["max"] == 7
    assert d["0.25"] == 2.5
    assert d["0.5"] == 4
    assert d["0.75"] == 5.5


def test_describe_list():
    s = Stats(range(1, 8))
    items = s.describe(format="list")
    assert isinstance(items, list)
    keys = [k for k, _ in items]
    assert keys == ["count", "mean", "std_dev", "mad", "min",
                    "0.25", "0.5", "0.75", "max"]


def test_describe_text():
    s = Stats(range(1, 8))
    out = s.describe(format="text")
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
    s = Stats(range(5))
    with pytest.raises(ValueError):
        s.describe(format="bogus")


def test_describe_custom_quantiles():
    s = Stats(range(1, 8))
    d = s.describe(quantiles=[0.1, 0.9])
    assert "0.1" in d
    assert "0.9" in d
    assert "0.25" not in d


def test_module_describe():
    out = describe(range(7), format="text")
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


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_module_functions():
    assert mean(range(42)) == 20.5
    assert median([2, 1, 3]) == 2
    assert mode([2, 1, 3, 1]) == 1
    assert variance(range(97)) == 784.0
    assert std_dev(range(97)) == 28.0
    assert median_abs_dev(range(97)) == 24.0
    assert round(rel_std_dev(range(97)), 3) == 0.583
    assert skewness(range(97)) == 0.0
    assert kurtosis(range(9)) == 1.99125
    assert trimean([2, 1, 3]) == 2.0
    assert iqr([1, 2, 3, 4, 5]) == 2


def test_module_functions_custom_default():
    assert mean([], default=5.0) == 5.0


# ---------------------------------------------------------------------------
# format_histogram_counts
# ---------------------------------------------------------------------------

def test_format_histogram_counts_basic():
    bin_counts = [(0.0, 5), (1.0, 8), (2.0, 11), (3.0, 5), (4.0, 2)]
    out = format_histogram_counts(bin_counts, width=30)
    lines = out.splitlines()
    assert len(lines) == 5
    # verify label padding and counts appear
    assert lines[0].startswith("0.0:  5 ")
    assert lines[2].startswith("2.0: 11 ")


def test_format_histogram_counts_format_bin():
    bin_counts = [(1.0, 3), (2.0, 6)]
    out = format_histogram_counts(
        bin_counts, width=40, format_bin=lambda v: "%dms" % v
    )
    lines = out.splitlines()
    assert lines[0].startswith("1ms: 3 ")
    assert lines[1].startswith("2ms: 6 ")


def test_format_histogram_counts_zero_count_uses_pipe():
    bin_counts = [(0.0, 0), (1.0, 4)]
    out = format_histogram_counts(bin_counts, width=40)
    lines = out.splitlines()
    # zero-count bin produces a '|' placeholder bar
    assert lines[0].rstrip().endswith("|")
    assert "#" in lines[1]
