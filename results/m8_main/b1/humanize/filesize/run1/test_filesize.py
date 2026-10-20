"""Tests for humanize.filesize."""

from __future__ import annotations

import pytest

from humanize.filesize import naturalsize, suffixes


class TestSuffixes:
    def test_decimal_suffixes(self):
        assert suffixes["decimal"] == (
            "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB",
        )

    def test_binary_suffixes(self):
        assert suffixes["binary"] == (
            "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB",
        )

    def test_gnu_suffixes(self):
        assert suffixes["gnu"] == "KMGTPEZYRQ"


class TestNaturalsizeDecimal:
    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_small_bytes(self):
        assert naturalsize(300) == "300 Bytes"

    def test_just_below_base(self):
        assert naturalsize(999) == "999 Bytes"

    def test_exactly_kilo(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_megabytes(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_gigabytes(self):
        assert naturalsize(3 * 10**9) == "3.0 GB"

    def test_negative_kilo(self):
        assert naturalsize(-1000) == "-1.0 kB"

    def test_negative_megabytes(self):
        assert naturalsize(-3000000) == "-3.0 MB"

    def test_string_input(self):
        assert naturalsize("3000000") == "3.0 MB"

    def test_float_input(self):
        assert naturalsize(3000.0) == "3.0 kB"

    def test_rounding_up_to_next_suffix(self):
        # 999999 bytes is 999.999 kB, which rounds to 1000.0 -> step up to MB
        assert naturalsize(999999) == "1.0 MB"

    def test_very_large(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_huge(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_custom_format(self):
        assert naturalsize(3000, format="%.3f") == "3.000 kB"


class TestNaturalsizeBinary:
    def test_small_bytes(self):
        assert naturalsize(300, binary=True) == "300 Bytes"

    def test_one_byte_binary(self):
        # abs_bytes == 1 branch is taken regardless of binary
        assert naturalsize(1, binary=True) == "1 Byte"

    def test_kib(self):
        assert naturalsize(3000, binary=True) == "2.9 KiB"

    def test_exactly_kib(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_mib(self):
        assert naturalsize(2 * 1024**2, binary=True) == "2.0 MiB"

    def test_negative_binary(self):
        assert naturalsize(-4096, binary=True) == "-4.0 KiB"

    def test_rounding_up_binary(self):
        # 1024**2 - 1 bytes = 1023.999 KiB -> 1024.0 -> step to MiB
        assert naturalsize(1024**2 - 1, binary=True) == "1.0 MiB"

    def test_huge_binary(self):
        assert naturalsize(1024**10 * 5, binary=True) == "5.0 QiB"


class TestNaturalsizeGNU:
    def test_zero(self):
        # gnu does not use "1 Byte" special case; small is "0B"
        assert naturalsize(0, gnu=True) == "0B"

    def test_small_bytes(self):
        assert naturalsize(300, False, True) == "300B"

    def test_one_byte(self):
        # gnu mode: no "1 Byte" special case (not gnu branch), falls to < base
        assert naturalsize(1, False, True) == "1B"

    def test_kilo(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"

    def test_exactly_kilo(self):
        assert naturalsize(1024, False, True) == "1.0K"

    def test_mega(self):
        assert naturalsize(1024**2, False, True) == "1.0M"

    def test_binary_ignored(self):
        # In gnu mode the binary argument is ignored
        assert naturalsize(1024, True, True) == "1.0K"

    def test_negative(self):
        assert naturalsize(-1024, False, True) == "-1.0K"

    def test_no_space(self):
        result = naturalsize(2048, False, True)
        assert " " not in result
        assert result == "2.0K"

    def test_rounding_up(self):
        assert naturalsize(1024**2 - 1, False, True) == "1.0M"


class TestNaturalsizeEdgeCases:
    def test_bool_true(self):
        # bool is int subclass; True == 1
        assert naturalsize(True) == "1 Byte"

    def test_bool_false(self):
        assert naturalsize(False) == "0 Bytes"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            naturalsize(None)  # type: ignore[arg-type]

    def test_abs_one_is_byte(self):
        # triggers the abs_bytes == 1 branch for both +1 and -1
        assert naturalsize(1) == "1 Byte"
        assert naturalsize(-1) == "-1 Byte"

    def test_just_under_base_decimal(self):
        assert naturalsize(999.0) == "999 Bytes"

    def test_custom_format_used(self):
        assert naturalsize(1500, format="%.0f") == "2 kB"
