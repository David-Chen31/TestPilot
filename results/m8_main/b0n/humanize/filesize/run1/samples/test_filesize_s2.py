"""Tests for humanize.filesize.naturalsize."""

from __future__ import annotations

import pytest

from humanize.filesize import naturalsize


class TestZeroAndSmallBytes:
    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_zero_with_format(self):
        assert naturalsize(0, format="%.3f") == "0 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_small_bytes(self):
        assert naturalsize(300) == "300 Bytes"

    def test_small_bytes_negative(self):
        assert naturalsize(-300) == "-300 Bytes"

    def test_just_below_base_decimal(self):
        assert naturalsize(999) == "999 Bytes"

    def test_just_below_base_binary(self):
        assert naturalsize(1023, binary=True) == "1023 Bytes"

    def test_zero_gnu(self):
        assert naturalsize(0, gnu=True) == "0B"

    def test_small_bytes_gnu(self):
        assert naturalsize(300, gnu=True) == "300B"

    def test_one_byte_gnu(self):
        # gnu mode never uses the singular "Byte" form
        assert naturalsize(1, gnu=True) == "1B"

    def test_negative_one_byte_gnu(self):
        assert naturalsize(-1, gnu=True) == "-1B"


class TestDecimal:
    def test_kilobytes(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_three_million(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_kilobyte_boundary(self):
        assert naturalsize(1000) == "1.0 kB"
        assert naturalsize(999) == "999 Bytes"

    def test_megabyte(self):
        assert naturalsize(1000**2) == "1.0 MB"

    def test_gigabyte(self):
        assert naturalsize(1000**3) == "1.0 GB"

    def test_terabyte(self):
        assert naturalsize(1000**4) == "1.0 TB"

    def test_fractional(self):
        assert naturalsize(1500) == "1.5 kB"

    def test_rounding_promotes_suffix(self):
        # 999999 -> 999.999 kB -> rounds to "1000.0 kB", which is promoted to MB
        assert naturalsize(999999) == "1.0 MB"

    def test_rounding_promotes_with_custom_format(self):
        # 999999 formatted with %.3f as kB gives 999.999 -> promoted to MB
        assert naturalsize(999999, format="%.3f") == "1.000 MB"

    def test_negative(self):
        assert naturalsize(-3000) == "-3.0 kB"

    def test_negative_rounding_promotes(self):
        # abs-based promotion also applies to negative numbers
        assert naturalsize(-999999) == "-1.0 MB"


class TestBinary:
    def test_kibibyte(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_binary_decimal_input(self):
        # 3000 bytes in binary is 2.929... KiB
        assert naturalsize(3000, binary=True) == "2.9 KiB"

    def test_mebibyte(self):
        assert naturalsize(1024**2, binary=True) == "1.0 MiB"

    def test_gibibyte(self):
        assert naturalsize(1024**3, binary=True) == "1.0 GiB"

    def test_negative_binary(self):
        assert naturalsize(-4096, binary=True) == "-4.0 KiB"

    def test_binary_rounding_promotes(self):
        # 1024**2 - 1 bytes -> 1023.999 KiB -> rounds up to "1024.0 KiB" -> promote
        assert naturalsize(1024**2 - 1, binary=True) == "1.0 MiB"

    def test_custom_format(self):
        assert naturalsize(1024, binary=True, format="%.2f") == "1.00 KiB"


class TestGNU:
    def test_small(self):
        assert naturalsize(300, gnu=True) == "300B"

    def test_kilobyte(self):
        # GNU uses binary (1024) base even though suffixes differ
        assert naturalsize(3000, gnu=True) == "2.9K"

    def test_explicit_kilobyte(self):
        assert naturalsize(1024, gnu=True) == "1.0K"

    def test_custom_format(self):
        assert naturalsize(3000, gnu=True, format="%.3f") == "2.930K"

    def test_megabyte(self):
        assert naturalsize(1024**2, gnu=True) == "1.0M"

    def test_gnu_ignores_binary(self):
        assert naturalsize(3000, binary=True, gnu=True) == "2.9K"

    def test_no_space_in_gnu(self):
        result = naturalsize(1024, gnu=True)
        assert " " not in result


class TestLargeMagnitudes:
    def test_ronnabyte(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_quettabyte(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_beyond_max_suffix(self):
        # when exp would exceed available suffixes, it is clamped to len(suffix)
        result = naturalsize(10**60)
        assert result.endswith("QB")

    def test_binary_large(self):
        assert naturalsize(1024**9) == "1.0 RiB"


class TestInputTypes:
    def test_integer(self):
        assert naturalsize(5000) == "5.0 kB"

    def test_float(self):
        assert naturalsize(5000.0) == "5.0 kB"

    def test_string(self):
        assert naturalsize("3000000") == "3.0 MB"

    def test_string_small(self):
        assert naturalsize("300") == "300 Bytes"

    def test_string_with_float(self):
        assert naturalsize("5000.0") == "5.0 kB"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")


class TestCustomFormat:
    def test_integer_format(self):
        assert naturalsize(3000000, format="%.0f") == "3 MB"

    def test_precision(self):
        assert naturalsize(1234567, format="%.2f") == "1.23 MB"

    def test_no_format_promotion(self):
        # With %.0f, 999999 kB -> "1000" which is >= base, promoted
        assert naturalsize(999999, format="%.0f") == "1 MB"


class TestDefaultArgumentsAndKeywordArgs:
    def test_all_keyword(self):
        assert naturalsize(value=3000, binary=False, gnu=False, format="%.1f") == "3.0 kB"

    def test_positional_binary(self):
        assert naturalsize(1024, True) == "1.0 KiB"

    def test_positional_gnu(self):
        assert naturalsize(300, False, True) == "300B"
