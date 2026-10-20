"""Tests for humanize.filesize.naturalsize."""

import math

import pytest

from humanize.filesize import naturalsize, suffixes


class TestBasicDecimal:
    def test_examples(self):
        assert naturalsize(3000000) == "3.0 MB"
        assert naturalsize(300) == "300 Bytes"
        assert naturalsize(1000) == "1.0 kB"
        assert naturalsize(1500) == "1.5 kB"

    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_negative_zero(self):
        assert naturalsize(-0.0) == "0 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_below_base(self):
        assert naturalsize(999) == "999 Bytes"

    def test_at_base(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_float_input(self):
        assert naturalsize(3000.0) == "3.0 kB"

    def test_string_input(self):
        assert naturalsize("3000") == "3.0 kB"

    def test_negative(self):
        assert naturalsize(-3000000) == "-3.0 MB"

    def test_custom_format(self):
        assert naturalsize(3000, format="%.3f") == "3.000 kB"

    def test_default_format(self):
        assert naturalsize(1234) == "1.2 kB"

    def test_truncation_below_base(self):
        # int() truncates for bytes
        assert naturalsize(999.9) == "999 Bytes"


class TestBinary:
    def test_one_byte(self):
        assert naturalsize(1, binary=True) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1, binary=True) == "-1 Byte"

    def test_below_base(self):
        assert naturalsize(1000, binary=True) == "1000 Bytes"

    def test_at_base(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_3000(self):
        assert naturalsize(3000, binary=True) == "2.9 KiB"

    def test_docstring_example(self):
        assert naturalsize(-4096, True) == "-4.0 KiB"

    def test_mebibyte(self):
        assert naturalsize(1024 * 1024, binary=True) == "1.0 MiB"


class TestGNU:
    def test_small(self):
        assert naturalsize(300, False, True) == "300B"

    def test_kilo(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"

    def test_zero(self):
        assert naturalsize(0, False, True) == "0B"

    def test_one_byte(self):
        # gnu mode has no special "Byte" singular handling
        assert naturalsize(1, False, True) == "1B"

    def test_negative_small(self):
        assert naturalsize(-300, False, True) == "-300B"

    def test_gnu_ignores_binary(self):
        assert naturalsize(3000, True, True) == "2.9K"

    def test_mega(self):
        assert naturalsize(1024 * 1024, False, True) == "1.0M"


class TestSuffixEscalation:
    def test_rounding_pushes_to_next_suffix(self):
        # 999999 is 999.999 kB, rounds to 1000.0 kB -> becomes 1.0 MB
        assert naturalsize(999999) == "1.0 MB"

    def test_no_escalation_when_below(self):
        assert naturalsize(999949) == "999.9 kB"


class TestLargeValues:
    def test_docstring_ronnabytes(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_docstring_quettabytes(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_binary_large(self):
        assert naturalsize(2**80, binary=True) == "1.0 YiB"

    def test_beyond_max_suffix_decimal(self):
        # exp is clamped to len(suffix)
        result = naturalsize(10**60)
        assert result.endswith("QB")
        expected = "%.1f" % (10**60 / (1000**10)) + " QB"
        assert result == expected

    def test_beyond_max_suffix_binary(self):
        result = naturalsize(10**60, binary=True)
        assert result.endswith("QiB")


class TestIntConversion:
    def test_negative_below_base_truncates(self):
        # int() truncates toward zero
        assert naturalsize(-999.9) == "-999 Bytes"

    def test_negative_one_singular(self):
        assert naturalsize(-1.0) == "-1 Byte"


class TestSuffixesStructure:
    def test_decimal_suffixes(self):
        assert suffixes["decimal"][:2] == ("kB", "MB")
        assert len(suffixes["decimal"]) == 10

    def test_binary_suffixes(self):
        assert suffixes["binary"][:2] == ("KiB", "MiB")
        assert len(suffixes["binary"]) == 10

    def test_gnu_suffixes(self):
        assert suffixes["gnu"] == "KMGTPEZYRQ"


class TestErrors:
    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")

    def test_none_raises(self):
        with pytest.raises(TypeError):
            naturalsize(None)
