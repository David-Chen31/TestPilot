import pytest

from humanize.filesize import naturalsize, suffixes


class TestSuffixesConstants:
    def test_decimal_suffixes(self):
        assert suffixes["decimal"][0] == "kB"
        assert suffixes["decimal"][1] == "MB"
        assert suffixes["decimal"][2] == "GB"
        assert suffixes["decimal"][-1] == "QB"
        assert len(suffixes["decimal"]) == 10

    def test_binary_suffixes(self):
        assert suffixes["binary"][0] == "KiB"
        assert suffixes["binary"][1] == "MiB"
        assert suffixes["binary"][-1] == "QiB"
        assert len(suffixes["binary"]) == 10

    def test_gnu_suffixes(self):
        assert suffixes["gnu"] == "KMGTPEZYRQ"


class TestByteAndSmallValues:
    def test_single_byte_singular(self):
        assert naturalsize(1) == "1 Byte"

    def test_single_byte_singular_negative(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_small_bytes_plural(self):
        assert naturalsize(300) == "300 Bytes"

    def test_negative_small_bytes(self):
        assert naturalsize(-300) == "-300 Bytes"

    def test_just_below_base(self):
        assert naturalsize(999) == "999 Bytes"

    def test_just_above_negative_base(self):
        assert naturalsize(-999) == "-999 Bytes"

    def test_float_small(self):
        assert naturalsize(1.5) == "1 Bytes"

    def test_float_truncation(self):
        assert naturalsize(500.9) == "500 Bytes"


class TestDecimalMode:
    def test_kilobytes(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_megabytes(self):
        assert naturalsize(3000000) == "3.0 MB"

    def test_gigabytes(self):
        assert naturalsize(10**9) == "1.0 GB"

    def test_no_space_in_gnu_but_space_in_decimal(self):
        # ensure space is present in decimal mode
        assert " " in naturalsize(1000)

    def test_negative_kilobytes(self):
        assert naturalsize(-4096) == "-4.1 kB"

    def test_rounding_rollover_to_megabyte(self):
        # 999999 bytes = 999.999 kB, rounds to 1000.0 -> should step up
        assert naturalsize(999999) == "1.0 MB"

    def test_rollover_boundary_exactly_base(self):
        assert naturalsize(1000000) == "1.0 MB"

    def test_large_value_rb(self):
        assert naturalsize(10**28) == "10.0 RB"

    def test_huge_value_qb(self):
        assert naturalsize(10**34 * 3) == "30000.0 QB"

    def test_max_exponent_capped(self):
        # exponent capped at len(suffix), so huge numbers stay at last suffix
        result = naturalsize(10**60)
        assert result.endswith("QB")


class TestBinaryMode:
    def test_binary_kib(self):
        assert naturalsize(3000, True) == "2.9 KiB"

    def test_binary_negative(self):
        assert naturalsize(-4096, True) == "-4.0 KiB"

    def test_binary_boundary_1024(self):
        assert naturalsize(1024, True) == "1.0 KiB"

    def test_binary_rollover(self):
        # 1023.999 KiB rounds up -> 1.0 MiB
        assert naturalsize(1024 * 1024 - 1, True) == "1.0 MiB"

    def test_binary_boundary_just_below(self):
        assert naturalsize(1023, True) == "1023 Bytes"


class TestGnuMode:
    def test_gnu_small(self):
        assert naturalsize(300, False, True) == "300B"

    def test_gnu_kilobytes(self):
        assert naturalsize(3000, False, True) == "2.9K"

    def test_gnu_no_space(self):
        assert " " not in naturalsize(3000, False, True)

    def test_gnu_single_byte_not_singular(self):
        # gnu ignores the singular "Byte" branch
        assert naturalsize(1, False, True) == "1B"

    def test_gnu_negative_single_byte(self):
        assert naturalsize(-1, False, True) == "-1B"

    def test_gnu_zero(self):
        assert naturalsize(0, False, True) == "0B"

    def test_gnu_ignores_binary(self):
        # binary=True but gnu=True -> gnu used
        assert naturalsize(3000, True, True) == "2.9K"

    def test_gnu_uses_binary_base(self):
        assert naturalsize(1024, False, True) == "1.0K"

    def test_gnu_custom_format(self):
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"


class TestCustomFormat:
    def test_custom_decimal(self):
        assert naturalsize(3000, format="%.3f") == "3.000 kB"

    def test_custom_format_zero_decimals(self):
        assert naturalsize(3000, format="%.0f") == "3 kB"

    def test_custom_format_binary(self):
        assert naturalsize(2048, True, format="%.2f") == "2.00 KiB"


class TestStringInput:
    def test_string_integer(self):
        assert naturalsize("3000") == "3.0 kB"

    def test_string_float(self):
        assert naturalsize("3000.0") == "3.0 kB"

    def test_string_small(self):
        assert naturalsize("300") == "300 Bytes"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")

    def test_invalid_string_raises_gnu(self):
        with pytest.raises(ValueError):
            naturalsize("abc", False, True)


class TestFloatInput:
    def test_float_value(self):
        assert naturalsize(1500.0) == "1.5 kB"

    def test_float_one(self):
        assert naturalsize(1.0) == "1 Byte"

    def test_float_negative_one(self):
        assert naturalsize(-1.0) == "-1 Byte"
