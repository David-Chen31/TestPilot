"""Tests for humanize.filesize."""

from __future__ import annotations

import inspect

import pytest

from humanize.filesize import naturalsize, suffixes


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


class TestNaturalsizeSignature:
    def test_default_format_is_lowercase_f(self):
        # The documented (and public) default format is "%.1f".
        # Assert on the declared default so a change to "%.1F" is caught
        # even on CPython where %F happens to behave like %f.
        sig = inspect.signature(naturalsize)
        assert sig.parameters["format"].default == "%.1f"

    def test_default_flags(self):
        sig = inspect.signature(naturalsize)
        assert sig.parameters["binary"].default is False
        assert sig.parameters["gnu"].default is False

    def test_call_with_no_format_matches_explicit_default(self):
        # Behavioral companion: omitting `format` matches passing "%.1f".
        assert naturalsize(1500) == naturalsize(1500, format="%.1f")
        assert naturalsize(3000000) == naturalsize(3000000, format="%.1f")


class TestNaturalsizeBasic:
    def test_docstring_examples(self):
        assert naturalsize(3000000) == "3.0 MB"
        assert naturalsize(300, False, True) == "300B"
        assert naturalsize(3000, False, True) == "2.9K"
        assert naturalsize(3000, False, True, "%.3f") == "2.930K"
        assert naturalsize(3000, True) == "2.9 KiB"
        assert naturalsize(10**28) == "10.0 RB"
        assert naturalsize(10**34 * 3) == "30000.0 QB"
        assert naturalsize(-4096, True) == "-4.0 KiB"

    def test_zero(self):
        assert naturalsize(0) == "0 Bytes"

    def test_single_byte(self):
        assert naturalsize(1) == "1 Byte"

    def test_negative_single_byte(self):
        assert naturalsize(-1) == "-1 Byte"

    def test_decimal_below_base(self):
        assert naturalsize(999) == "999 Bytes"

    def test_decimal_at_base(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_negative_bytes(self):
        assert naturalsize(-1000) == "-1.0 kB"


class TestDefaultFormat:
    def test_default_format_lowercase_f(self):
        # The default format string must use lowercase "f" ("%.1f").
        # With "%.1F" the output would be uppercase ("1.0 KB") only on some
        # platforms; on CPython %F is accepted as f, so use an exact-case
        # assertion at a boundary where a difference would surface.
        assert naturalsize(1500) == "1.5 kB"

    def test_default_format_single_decimal(self):
        assert naturalsize(1234) == "1.2 kB"

    def test_default_format_truncation_rounding(self):
        assert naturalsize(1999) == "2.0 kB"

    def test_default_format_mid_range(self):
        assert naturalsize(5555) == "5.6 kB"

    def test_default_format_exact(self):
        assert naturalsize(2500) == "2.5 kB"

    def test_default_format_uppercase_f_equivalent(self):
        # On CPython %F behaves like %f; this documents that the default
        # format still yields a lowercase suffix "kB" (not "KB").
        result = naturalsize(1500)
        assert result.endswith("kB")
        assert not result.endswith("KB")


class TestNaturalsizeDecimal:
    def test_kb(self):
        assert naturalsize(1000) == "1.0 kB"

    def test_mb(self):
        assert naturalsize(1000**2) == "1.0 MB"

    def test_gb(self):
        assert naturalsize(1000**3) == "1.0 GB"

    def test_tb(self):
        assert naturalsize(1000**4) == "1.0 TB"

    def test_pb(self):
        assert naturalsize(1000**5) == "1.0 PB"

    def test_eb(self):
        assert naturalsize(1000**6) == "1.0 EB"

    def test_zb(self):
        assert naturalsize(1000**7) == "1.0 ZB"

    def test_yb(self):
        assert naturalsize(1000**8) == "1.0 YB"

    def test_rb(self):
        assert naturalsize(1000**9) == "1.0 RB"

    def test_qb(self):
        assert naturalsize(1000**10) == "1.0 QB"

    def test_huge_value_caps_at_qb(self):
        # Exponent is capped at len(suffix) == 10 -> QB
        assert naturalsize(1000**12) == "1000000.0 QB"


class TestNaturalsizeBinary:
    def test_kib(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"

    def test_mib(self):
        assert naturalsize(1024**2, binary=True) == "1.0 MiB"

    def test_gib(self):
        assert naturalsize(1024**3, binary=True) == "1.0 GiB"

    def test_tib(self):
        assert naturalsize(1024**4, binary=True) == "1.0 TiB"

    def test_pib(self):
        assert naturalsize(1024**5, binary=True) == "1.0 PiB"

    def test_eib(self):
        assert naturalsize(1024**6, binary=True) == "1.0 EiB"

    def test_zib(self):
        assert naturalsize(1024**7, binary=True) == "1.0 ZiB"

    def test_yib(self):
        assert naturalsize(1024**8, binary=True) == "1.0 YiB"

    def test_rib(self):
        assert naturalsize(1024**9, binary=True) == "1.0 RiB"

    def test_qib(self):
        assert naturalsize(1024**10, binary=True) == "1.0 QiB"

    def test_below_base_uses_bytes(self):
        assert naturalsize(1023, binary=True) == "1023 Bytes"

    def test_at_base(self):
        assert naturalsize(1024, binary=True) == "1.0 KiB"


class TestNaturalsizeGnu:
    def test_zero_gnu(self):
        assert naturalsize(0, gnu=True) == "0B"

    def test_one_gnu(self):
        # GNU mode never returns "Byte"/"Bytes", always "B" no space
        assert naturalsize(1, gnu=True) == "1B"

    def test_below_base_gnu(self):
        assert naturalsize(999, gnu=True) == "999B"

    def test_at_base_gnu(self):
        assert naturalsize(1024, gnu=True) == "1.0K"

    def test_k_gnu(self):
        assert naturalsize(1024, gnu=True) == "1.0K"

    def test_m_gnu(self):
        assert naturalsize(1024**2, gnu=True) == "1.0M"

    def test_g_gnu(self):
        assert naturalsize(1024**3, gnu=True) == "1.0G"

    def test_t_gnu(self):
        assert naturalsize(1024**4, gnu=True) == "1.0T"

    def test_p_gnu(self):
        assert naturalsize(1024**5, gnu=True) == "1.0P"

    def test_e_gnu(self):
        assert naturalsize(1024**6, gnu=True) == "1.0E"

    def test_z_gnu(self):
        assert naturalsize(1024**7, gnu=True) == "1.0Z"

    def test_y_gnu(self):
        assert naturalsize(1024**8, gnu=True) == "1.0Y"

    def test_r_gnu(self):
        assert naturalsize(1024**9, gnu=True) == "1.0R"

    def test_q_gnu(self):
        assert naturalsize(1024**10, gnu=True) == "1.0Q"

    def test_gnu_ignores_binary(self):
        assert naturalsize(1024, binary=True, gnu=True) == "1.0K"

    def test_negative_gnu(self):
        assert naturalsize(-1024, gnu=True) == "-1.0K"


class TestNaturalsizeFormat:
    def test_custom_format(self):
        assert naturalsize(1500, format="%.2f") == "1.50 kB"

    def test_custom_format_no_decimals(self):
        assert naturalsize(1500, format="%.0f") == "2 kB"

    def test_custom_format_gnu(self):
        assert naturalsize(1536, gnu=True, format="%.2f") == "1.50K"


class TestNaturalsizeRounding:
    def test_rounding_steps_up(self):
        # 999999 is 999.999 kB; rounds to 1000.0 -> steps up to MB
        assert naturalsize(999999) == "1.0 MB"

    def test_rounding_at_higher_boundary(self):
        # 999 >= base -> stays kB (not >= base after formatting)
        assert naturalsize(999) == "999 Bytes"

    def test_rounding_near_boundary_keeps_suffix(self):
        # 99994 -> 99.994 kB, formats to 100.0 kB (below base) stays kB
        assert naturalsize(99994) == "100.0 kB"

    def test_rounding_exact_boundary(self):
        # 999950 / 1000 = 999.95 -> "%.1f" -> "1000.0" -> steps up to MB
        assert naturalsize(999950) == "1.0 MB"


class TestNaturalsizeStringInput:
    def test_string_integer(self):
        assert naturalsize("1000") == "1.0 kB"

    def test_string_float(self):
        assert naturalsize("1500.0") == "1.5 kB"

    def test_string_negative(self):
        assert naturalsize("-1000") == "-1.0 kB"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            naturalsize("not a number")


class TestNaturalsizeFloatInput:
    def test_float_input(self):
        assert naturalsize(1500.0) == "1.5 kB"

    def test_float_single_byte(self):
        assert naturalsize(1.0) == "1 Byte"
