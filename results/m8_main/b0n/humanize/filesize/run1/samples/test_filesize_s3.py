"""Tests for humanize.filesize."""

from __future__ import annotations

import pytest

from humanize.filesize import naturalsize, suffixes


class TestDecimal:
    """Default decimal (1000-based) behavior."""

    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_below_base(self):
        assert naturalsize(999) == "999 Bytes"

    def test_negative_below_base(self):
        assert naturalsize(-999) == "-999 Bytes"

    def test_kilo(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_mega(self):
        assert naturalsize(3_000_000) == "3.0 MB"

    def test_giga(self):
        assert naturalsize(10**9) == "1.0 GB"

    def test_tera(self):
        assert naturalsize(10**12) == "1.0 TB"

    def test_peta(self):
        assert naturalsize(10**15) == "1.0 PB"

    def test_exa(self):
        assert naturalsize(10**18) == "1.0 EB"

    def test_zetta(self):
        assert naturalsize(10**21) == "1.0 ZB"

    def test_yotta(self):
        assert naturalsize(10**24) == "1.0 YB"

    def test_ronna(self):
        assert naturalsize(10**27) == "1.0 RB"

    def test_quetta(self):
        assert naturalsize(10**30) == "1.0 QB"

    def test_rounding_up_to_next_suffix_from_kilo(self):
        # 999999 bytes is 999.999 kB, formats to 1000.0 -> steps up to MB
        assert naturalsize(999_999) == "1.0 MB"

    def test_large_qb(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_negative_value(self):
        assert naturalsize(-1000) == "-1.0 kB"

    def test_float_value(self):
        assert naturalsize(1500.0) == "1.5 kB"

    def test_string_value(self):
        assert naturalsize("2000") == "2.0 kB"

    def test_float_string(self):
        assert naturalsize("1500.0") == "1.5 kB"


class TestBinary:
    """Binary (1024-based) behavior."""

    def test_one_kib(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_mib(self):
        assert naturalsize(1024**2, binary=True) == "1.0 MiB"

    def test_gib(self):
        assert naturalsize(1024**3, binary=True) == "1.0 GiB"

    def test_kibish_value(self):
        assert naturalsize(3000, binary=True) == "2.9 KiB"

    def test_negative(self):
        assert naturalsize(-4096, binary=True) == "-4.0 KiB"

    def test_one_byte_binary(self):
        assert naturalsize(1, binary=True) == "1 Byte"

    def test_below_1024(self):
        assert naturalsize(1023, binary=True) == "1023 Bytes"

    def test_rounding_up_to_next_binary_suffix(self):
        # 1024**2 - 1 bytes at binary base is ~1023.999 KiB, formats to 1024.0
        # -> steps up to MiB
        assert naturalsize(1024**2 - 1, binary=True) == "1.0 MiB"


class TestGnu:
    """GNU-style behavior (no decimal point suffix spacing, base 1024)."""

    def test_small(self):
        assert naturalsize(300, False, True) == "300B"

    def test_zero(self):
        assert naturalsize(0, gnu=True) == "0B"

    def test_one_byte(self):
        # GNU mode never returns "1 Byte"; returns "1B"
        assert naturalsize(1, gnu=True) == "1B"

    def test_kilo(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_mega(self):
        assert naturalsize(1024**2, gnu=True) == "1.0M"

    def test_giga(self):
        assert naturalsize(1024**3, gnu=True) == "1.0G"

    def test_negative(self):
        assert naturalsize(-3000, gnu=True) == "-2.9K"

    def test_binary_ignored(self):
        # gnu takes precedence over binary
        assert naturalsize(3000, binary=True, gnu=True) == "2.9K"


class TestCustomFormat:
    def test_decimal_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"

    def test_binary_custom_format(self):
        assert naturalsize(1536, binary=True, format="%.2f") == "1.50 KiB"

    def test_custom_format_no_decimals(self):
        assert naturalsize(1500, format="%.0f") == "2 kB"

    def test_rounding_step_up_with_custom_format(self):
        # Using integer round, 999999 -> 1000 kB -> step to MB
        assert naturalsize(999_999, format="%.0f") == "1 MB"


class TestSuffixes:
    def test_decimal_suffixes(self):
        assert suffixes["decimal"] == (
            "kB",
            "MB",
            "GB",
            "TB",
            "PB",
            "EB",
            "ZB",
            "YB",
            "RB",
            "QB",
        )

    def test_binary_suffixes(self):
        assert suffixes["binary"] == (
            "KiB",
            "MiB",
            "GiB",
            "TiB",
            "PiB",
            "EiB",
            "ZiB",
            "YiB",
            "RiB",
            "QiB",
        )

    def test_gnu_suffixes(self):
        assert suffixes["gnu"] == "KMGTPEZYRQ"


class TestErrors:
    def test_invalid_string(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")

    def test_none(self):
        with pytest.raises(TypeError):
            naturalsize(None)  # type: ignore[arg-type]
