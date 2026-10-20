"""Tests for humanize.filesize."""

from __future__ import annotations

import pytest

from humanize.filesize import naturalsize


class TestNaturalsizeDecimal:
    """Default (decimal) suffix behavior."""

    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_small_bytes(self):
        assert naturalsize(300) == "300 Bytes"

    def test_999_bytes(self):
        assert naturalsize(999) == "999 Bytes"

    def test_just_below_kb(self):
        assert naturalsize(999) == "999 Bytes"

    def test_one_kb(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_docstring_example_mb(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_negative_kb(self):
        assert naturalsize(-4096) == "-4.1 kB"

    def test_negative_binary(self):
        assert naturalsize(-4096, True) == "-4.0 KiB"

    def test_float_value(self):
        assert naturalsize(1500.0) == "1.5 kB"

    def test_string_value(self):
        assert naturalsize("3000000") == "3.0 MB"

    def test_rounded_up_to_next_suffix(self):
        # 999999 -> 999.999 kB rounds to 1000.0 kB, promoted to MB
        assert naturalsize(999999) == "1.0 MB"

    def test_large_rb(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_huge_qb(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"


class TestNaturalsizeBinary:
    """Binary suffix behavior."""

    def test_one_byte_still_single(self):
        assert naturalsize(1, binary=True) == "1 Byte"

    def test_zero(self):
        assert naturalsize(0, binary=True) == "0 Bytes"

    def test_just_below_kib(self):
        assert naturalsize(1023, binary=True) == "1023 Bytes"

    def test_one_kib(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_docstring_example(self):
        assert naturalsize(3000, True) == "2.9 KiB"

    def test_mib(self):
        assert naturalsize(1024 * 1024, binary=True) == "1.0 MiB"


class TestNaturalsizeGnu:
    """GNU (ls -sh style) behavior."""

    def test_small_no_suffix(self):
        assert naturalsize(300, False, True) == "300B"

    def test_one_byte_gnu(self):
        # gnu mode never uses the "1 Byte" special case
        assert naturalsize(1, False, True) == "1B"

    def test_zero_gnu(self):
        assert naturalsize(0, False, True) == "0B"

    def test_k(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"

    def test_gnu_ignores_binary(self):
        # binary ignored in gnu mode; base 1024 used
        assert naturalsize(1024, True, True) == "1.0K"

    def test_negative_gnu(self):
        assert naturalsize(-300, False, True) == "-300B"


class TestNaturalsizeFormat:
    """Custom format argument."""

    def test_integer_format(self):
        assert naturalsize(3000000, format="%d") == "3 MB"

    def test_no_decimals(self):
        assert naturalsize(1500, format="%.0f") == "2 kB"


class TestNaturalsizeEdgeCases:
    """Edge cases and boundary values."""

    def test_boundary_exact_base(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_boundary_binary_exact(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_below_base_binary(self):
        assert naturalsize(1023) == "1023 Bytes"

    def test_negative_small(self):
        assert naturalsize(-999) == "-999 Bytes"

    def test_float_negative(self):
        assert naturalsize(-1.0) == "-1 Byte"

    def test_string_negative(self):
        assert naturalsize("-4096", True) == "-4.0 KiB"

    def test_huge_value_last_suffix(self):
        # Beyond available suffixes stays on last one
        result = naturalsize(10**100)
        assert result.endswith("QB")

    @pytest.mark.parametrize("value", [1, 2, 10, 100, 999])
    def test_small_values_use_bytes_suffix(self, value):
        result = naturalsize(value)
        assert result.endswith("Bytes") or result.endswith("Byte")
