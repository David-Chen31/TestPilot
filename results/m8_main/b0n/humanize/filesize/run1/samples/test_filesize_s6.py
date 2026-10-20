"""Tests for humanize.filesize."""

import pytest

from humanize.filesize import naturalsize


class TestNaturalsizeDecimal:
    """Tests for decimal (default) suffix behavior."""

    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_minus_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_small_bytes(self):
        assert naturalsize(2) == "2 Bytes"
        assert naturalsize(300) == "300 Bytes"
        assert naturalsize(999) == "999 Bytes"

    def test_negative_small_bytes(self):
        assert naturalsize(-300) == "-300 Bytes"

    def test_exactly_base(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_below_base(self):
        assert naturalsize(999.9) == "999 Bytes"

    def test_one_kilobyte(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_megabyte(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_gigabyte(self):
        assert naturalsize(10**9) == "1.0 GB"

    def test_terabyte(self):
        assert naturalsize(10**12) == "1.0 TB"

    def test_negative_decimal(self):
        assert naturalsize(-4096) == "-4.1 kB"

    def test_large_rb(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_very_large_qb(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_rounding_steps_up_suffix(self):
        # 999999 bytes = 999.999 kB, which rounds to 1000.0 kB -> step to MB
        assert naturalsize(999999) == "1.0 MB"

    def test_rounding_no_step_when_max_suffix(self):
        # Near max suffix, no larger suffix exists to step up to
        result = naturalsize(10**34)
        assert result.endswith("QB")

    def test_negative_rounding_steps_up_suffix(self):
        assert naturalsize(-999999) == "-1.0 MB"


class TestNaturalsizeBinary:
    """Tests for binary suffix behavior."""

    def test_one_byte_binary(self):
        assert naturalsize(1, True) == "1 Byte"

    def test_small_bytes_binary(self):
        assert naturalsize(300, True) == "300 Bytes"

    def test_exactly_1024(self):
        assert naturalsize(1024, True) == "1.0 KiB"

    def test_below_1024(self):
        assert naturalsize(1023, True) == "1023 Bytes"

    def test_kibibyte(self):
        assert naturalsize(3000, True) == "2.9 KiB"

    def test_mebibyte(self):
        assert naturalsize(1024**2, True) == "1.0 MiB"

    def test_gibibyte(self):
        assert naturalsize(1024**3, True) == "1.0 GiB"

    def test_negative_binary(self):
        assert naturalsize(-4096, True) == "-4.0 KiB"

    def test_zero_binary(self):
        assert naturalsize(0, True) == "0 Bytes"

    def test_rounding_steps_up_binary(self):
        # 1024**2 - 1 bytes is just under 1 MiB, formats to 1024.0 KiB -> 1.0 MiB
        assert naturalsize(1024**2 - 1, True) == "1.0 MiB"


class TestNaturalsizeGnu:
    """Tests for GNU-style suffix behavior."""

    def test_one_byte_gnu(self):
        # gnu does not special-case "Byte"
        assert naturalsize(1, gnu=True) == "1B"

    def test_small_bytes_gnu(self):
        assert naturalsize(300, False, True) == "300B"

    def test_zero_gnu(self):
        assert naturalsize(0, gnu=True) == "0B"

    def test_kilobyte_gnu(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_kilobyte_gnu_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"

    def test_exactly_1024_gnu(self):
        assert naturalsize(1024, gnu=True) == "1.0K"

    def test_megabyte_gnu(self):
        assert naturalsize(1024**2, gnu=True) == "1.0M"

    def test_gigabyte_gnu(self):
        assert naturalsize(1024**3, gnu=True) == "1.0G"

    def test_gnu_ignores_binary(self):
        assert naturalsize(1024, True, True) == "1.0K"

    def test_negative_gnu(self):
        assert naturalsize(-1024, gnu=True) == "-1.0K"

    def test_no_space_in_gnu(self):
        result = naturalsize(1024**2, gnu=True)
        assert " " not in result

    def test_rounding_steps_up_gnu(self):
        assert naturalsize(1024**2 - 1, gnu=True) == "1.0M"


class TestNaturalsizeFormat:
    """Tests for custom format strings."""

    def test_default_format(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_integer_format(self):
        assert naturalsize(1500, format="%.0f") == "2 kB"

    def test_three_decimals(self):
        assert naturalsize(3000, format="%.3f") == "3.000 kB"

    def test_no_decimal_format(self):
        assert naturalsize(1234567, format="%.0f") == "1 MB"

    def test_negative_format(self):
        assert naturalsize(-3000, format="%.2f") == "-3.00 kB"

    def test_custom_format_binary(self):
        assert naturalsize(3000, True, False, "%.3f") == "2.930 KiB"


class TestNaturalsizeStringInput:
    """Tests for string numeric input."""

    def test_string_integer(self):
        assert naturalsize("3000000") == "3.0 MB"

    def test_string_float(self):
        assert naturalsize("3000") == "3.0 kB"

    def test_string_negative(self):
        assert naturalsize("-4096", True) == "-4.0 KiB"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")


class TestNaturalsizeFloatInput:
    """Tests for float input values."""

    def test_float_input(self):
        assert naturalsize(3000.0) == "3.0 kB"

    def test_float_fractional(self):
        assert naturalsize(1.5) == "1 Bytes"

    def test_float_one(self):
        assert naturalsize(1.0) == "1 Byte"


class TestNaturalsizeEdgeCases:
    """Additional edge cases."""

    def test_negative_zero(self):
        assert naturalsize(-0.0) == "0 Bytes"

    def test_one_negative_binary(self):
        assert naturalsize(-1, True) == "-1 Byte"

    def test_exactly_base_binary(self):
        assert naturalsize(1024, True) == "1.0 KiB"

    def test_just_below_base(self):
        assert naturalsize(1023) == "1023 Bytes"

    def test_just_above_base(self):
        assert naturalsize(1001) == "1.0 kB"
