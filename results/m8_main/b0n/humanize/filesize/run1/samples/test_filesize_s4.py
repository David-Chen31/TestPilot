import pytest

from humanize.filesize import naturalsize


class TestNaturalsizeBasic:
    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_small_positive(self):
        assert naturalsize(300) == "300 Bytes"

    def test_small_negative(self):
        assert naturalsize(-300) == "-300 Bytes"

    def test_one_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_one_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_decimal_default(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_decimal_kb(self):
        assert naturalsize(3000) == "3.0 kB"


class TestNaturalsizeGnu:
    def test_gnu_below_base(self):
        assert naturalsize(300, False, True) == "300B"

    def test_gnu_one_byte(self):
        # gnu bypasses the "%d Byte" special case
        assert naturalsize(1, False, True) == "1B"

    def test_gnu_zero(self):
        assert naturalsize(0, False, True) == "0B"

    def test_gnu_negative_below_base(self):
        assert naturalsize(-300, False, True) == "-300B"

    def test_gnu_3000(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_gnu_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"

    def test_gnu_suffixes(self):
        assert naturalsize(1024, gnu=True) == "1.0K"
        assert naturalsize(1024**2, gnu=True) == "1.0M"
        assert naturalsize(1024**3, gnu=True) == "1.0G"

    def test_gnu_ignores_binary(self):
        assert naturalsize(3000, True, True) == "2.9K"


class TestNaturalsizeBinary:
    def test_binary_kib(self):
        assert naturalsize(3000, True) == "2.9 KiB"

    def test_binary_negative(self):
        assert naturalsize(-4096, True) == "-4.0 KiB"

    def test_binary_exact(self):
        assert naturalsize(1024, True) == "1.0 KiB"
        assert naturalsize(1024**2, True) == "1.0 MiB"
        assert naturalsize(1024**3, True) == "1.0 GiB"


class TestNaturalsizeLarge:
    def test_ronnabytes(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_quettabytes(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_suffix_cap(self):
        # beyond QB the exponent is capped at len(suffix)
        result = naturalsize(10**40)
        assert result.endswith("QB")

    def test_rounding_steps_up_suffix(self):
        # 999999 is 999.999 kB, which formats to "1000.0 kB"; steps up to MB
        assert naturalsize(999999) == "1.0 MB"

    def test_rounding_steps_up_suffix_binary(self):
        # Just under 1 MiB rounds to 1024.0 KiB -> steps up to MiB
        assert naturalsize(1024 * 1024 - 1, True) == "1.0 MiB"


class TestNaturalsizeString:
    def test_string_value(self):
        assert naturalsize("3000") == "3.0 kB"

    def test_string_negative(self):
        assert naturalsize("-3000") == "-3.0 kB"


class TestNaturalsizeFormat:
    def test_custom_format_integer(self):
        assert naturalsize(3000000, format="%d MB"[:0] + "%.0f") == "3 MB"

    def test_custom_format_zero_decimals(self):
        assert naturalsize(1500000, format="%.0f") == "2 MB"


class TestNaturalsizeExactBoundaries:
    def test_exactly_base(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_exactly_base_binary(self):
        assert naturalsize(1024, True) == "1.0 KiB"

    def test_just_below_base(self):
        assert naturalsize(999) == "999 Bytes"

    def test_just_below_base_binary(self):
        assert naturalsize(1023, True) == "1023 Bytes"
