import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert clamp(1.0, 0, 5) == 1.0
        assert clamp(3, 0, 5) == 3

    def test_below_lower(self):
        assert clamp(-1.0, 0, 5) == 0
        assert clamp(-100, -50, 50) == -50

    def test_above_upper(self):
        assert clamp(101.0, 0, 5) == 5
        assert clamp(123, upper=5) == 5

    def test_only_upper(self):
        assert clamp(10, upper=5) == 5
        assert clamp(-10, upper=5) == -10

    def test_only_lower(self):
        assert clamp(-10, lower=-5) == -5
        assert clamp(10, lower=-5) == 10

    def test_no_bounds(self):
        assert clamp(42) == 42
        assert clamp(-42) == -42

    def test_equal_bounds(self):
        assert clamp(5, 5, 5) == 5
        assert clamp(3, 5, 5) == 5
        assert clamp(7, 5, 5) == 5

    def test_lower_greater_than_upper_raises(self):
        with pytest.raises(ValueError):
            clamp(1, 10, 5)

    def test_returns_bound_object(self):
        # When clamped to a bound, the bound value itself is returned.
        lower = 0
        upper = 5
        assert clamp(-1, lower, upper) is lower
        assert clamp(10, lower, upper) is upper


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

class TestCeil:
    def test_plain(self):
        assert ceil(1.1) == 2
        assert ceil(1.0) == 1
        assert ceil(-1.1) == -1
        assert ceil(-1.0) == -1
        assert ceil(0) == 0
        assert ceil(5) == 5

    def test_with_options(self):
        options = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
        assert ceil(3.5, options=options) == 4
        assert ceil(4, options=options) == 4
        assert ceil(1.5, options=options) == 1.5
        assert ceil(0, options=options) == 1.5

    def test_options_unsorted(self):
        assert ceil(3.5, options=[10, 1.5, 4, 6]) == 4

    def test_options_generator(self):
        assert ceil(3.5, options=iter([1.5, 4, 6])) == 4

    def test_options_no_ceil_raises(self):
        with pytest.raises(ValueError):
            ceil(100, options=[1, 2, 3])

    def test_options_empty_raises(self):
        with pytest.raises(ValueError):
            ceil(0, options=[])

    def test_exact_match_returns_option(self):
        opt = 4
        assert ceil(4, options=[1, opt, 5]) is opt


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

class TestFloor:
    def test_plain(self):
        assert floor(1.1) == 1
        assert floor(1.0) == 1
        assert floor(-1.1) == -2
        assert floor(-1.0) == -1
        assert floor(0) == 0
        assert floor(5) == 5

    def test_with_options(self):
        options = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
        assert floor(3.5, options=options) == 2.5
        assert floor(2.5, options=options) == 2.5
        assert floor(50, options=options) == 50
        assert floor(1000, options=options) == 50

    def test_options_unsorted(self):
        assert floor(3.5, options=[10, 1.5, 2.5, 6]) == 2.5

    def test_options_generator(self):
        assert floor(3.5, options=iter([1.5, 2.5, 6])) == 2.5

    def test_options_no_floor_raises(self):
        with pytest.raises(ValueError):
            floor(0, options=[1, 2, 3])

    def test_options_empty_raises(self):
        with pytest.raises(ValueError):
            floor(0, options=[])

    def test_exact_match_returns_option(self):
        opt = 4
        assert floor(4, options=[1, opt, 5]) is opt


# ---------------------------------------------------------------------------
# Bits: construction
# ---------------------------------------------------------------------------

class TestBitsConstruction:
    def test_default(self):
        b = Bits()
        assert b.as_int() == 0
        assert b.len == 1

    def test_from_int(self):
        b = Bits(5)
        assert b.as_int() == 5
        assert b.len == 3
        assert b.as_bin() == '101'

    def test_from_int_with_len(self):
        b = Bits(5, 8)
        assert b.as_int() == 5
        assert b.len == 8
        assert b.as_bin() == '00000101'

    def test_from_int_zero(self):
        b = Bits(0)
        assert b.as_int() == 0
        assert b.len == 1
        assert b.as_bin() == '0'

    def test_from_int_zero_with_len(self):
        b = Bits(0, 4)
        assert b.as_int() == 0
        assert b.len == 4
        assert b.as_bin() == '0000'

    def test_from_bin_string(self):
        b = Bits('101')
        assert b.as_int() == 5
        assert b.len == 3

    def test_from_empty_bin_string(self):
        b = Bits('')
        assert b.as_int() == 0
        assert b.len == 0
        assert b.as_bin() == ''
        assert b.as_hex() == ''

    def test_from_hex_string(self):
        b = Bits('0x1F')
        assert b.as_int() == 0x1F
        assert b.len == 8
        assert b.as_hex() == '1F'

    def test_from_list(self):
        b = Bits([True, False, True])
        assert b.as_int() == 5
        assert b.len == 3
        assert b.as_list() == [True, False, True]

    def test_from_bytes(self):
        b = Bits(b'101')
        assert b.as_int() == 5
        assert b.len == 3

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            Bits(-1)

    def test_too_small_len_raises(self):
        with pytest.raises(ValueError):
            Bits(8, 3)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            Bits(1.5)
        with pytest.raises(TypeError):
            Bits(object())


# ---------------------------------------------------------------------------
# Bits: classmethod constructors
# ---------------------------------------------------------------------------

class TestBitsClassmethods:
    def test_from_list(self):
        b = Bits.from_list([1, 0, 1])
        assert b.as_int() == 5
        assert b.len == 3

    def test_from_bin(self):
        b = Bits.from_bin('101')
        assert b.as_int() == 5
        assert b.len == 3

    def test_from_hex(self):
        b = Bits.from_hex('1F')
        assert b.as_int() == 0x1F
        assert b.len == 8

    def test_from_hex_with_prefix(self):
        b = Bits.from_hex('0x1F')
        assert b.as_int() == 0x1F
        assert b.len == 8

    def test_from_hex_bytes(self):
        b = Bits.from_hex(b'1F')
        assert b.as_int() == 0x1F
        assert b.len == 8

    def test_from_hex_empty(self):
        b = Bits.from_hex('')
        assert b.as_int() == 0
        assert b.len == 0

    def test_from_int(self):
        b = Bits.from_int(5, 8)
        assert b.as_int() == 5
        assert b.len == 8

    def test_from_bytes(self):
        b = Bits.from_bytes(b'\x01\xf0')
        assert b.as_int() == 0x01F0
        assert b.len == 16


# ---------------------------------------------------------------------------
# Bits: conversions
# ---------------------------------------------------------------------------

class TestBitsConversions:
    def test_as_list(self):
        b = Bits('101')
        assert b.as_list() == [True, False, True]

    def test_as_bin(self):
        assert Bits(5).as_bin() == '101'
        assert Bits(5, 8).as_bin() == '00000101'

    def test_as_hex(self):
        assert Bits(5, 8).as_hex() == '05'
        assert Bits(0xFF, 8).as_hex() == 'FF'

    def test_as_hex_partial_byte(self):
        # 3 bits -> one byte -> 2 hex chars
        assert Bits(5, 3).as_hex() == '05'

    def test_as_int(self):
        assert Bits(5).as_int() == 5

    def test_as_bytes(self):
        b = Bits(0x01F0, 16)
        assert b.as_bytes() == b'\x01\xf0'

    def test_as_bytes_padded(self):
        b = Bits(5, 8)
        assert b.as_bytes() == b'\x05'


# ---------------------------------------------------------------------------
# Bits: indexing, len, equality, hash, repr
# ---------------------------------------------------------------------------

class TestBitsProtocol:
    def test_len(self):
        assert len(Bits('101')) == 3
        assert len(Bits('')) == 0

    def test_getitem(self):
        b = Bits('101')
        assert b[0] is True
        assert b[1] is False
        assert b[2] is True

    def test_getitem_negative(self):
        b = Bits('101')
        assert b[-1] is True
        assert b[-2] is False
        assert b[-3] is True

    def test_getitem_out_of_range(self):
        b = Bits('101')
        with pytest.raises(IndexError):
            b[3]
        with pytest.raises(IndexError):
            b[-4]

    def test_getitem_bad_type(self):
        b = Bits('101')
        with pytest.raises(TypeError):
            b['a']

    def test_getitem_slice(self):
        b = Bits('10110')
        assert b[1:4] == Bits('011')
        assert b[:2] == Bits('10')
        assert b[2:] == Bits('110')

    def test_eq(self):
        assert Bits('101') == Bits('101')
        assert Bits(5, 3) == Bits('101')
        assert not (Bits('101') == Bits('100'))
        assert not (Bits(5, 3) == Bits(5, 4))

    def test_eq_other_type(self):
        assert Bits('101') != '101'
        assert Bits('101') != 5

    def test_hash(self):
        assert hash(Bits('101')) == hash(5)

    def test_repr(self):
        assert repr(Bits('101')) == "Bits('101')"


# ---------------------------------------------------------------------------
# Bits: bitwise operators
# ---------------------------------------------------------------------------

class TestBitsOperators:
    def test_or(self):
        a = Bits('1010')
        b = Bits('0110')
        c = a | b
        assert c.as_int() == 0b1110
        assert c.len == 4

    def test_or_different_lengths(self):
        a = Bits('1')       # len 1
        b = Bits('0001')    # len 4
        c = a | b
        assert c.len == 4
        assert c.as_int() == 1

    def test_and(self):
        a = Bits('1010')
        b = Bits('0110')
        c = a & b
        assert c.as_int() == 0b0010
        assert c.len == 4

    def test_and_different_lengths(self):
        a = Bits('1')
        b = Bits('0001')
        c = a & b
        assert c.len == 4
        assert c.as_int() == 1

    def test_lshift(self):
        a = Bits('1')       # len 1, val 1
        b = a << 2
        assert b.as_int() == 4
        assert b.len == 3
        assert b.as_bin() == '100'

    def test_rshift(self):
        a = Bits('100')     # len 3, val 4
        b = a >> 2
        assert b.as_int() == 1
        assert b.len == 1
        assert b.as_bin() == '1'

    def test_or_other_type_not_implemented(self):
        with pytest.raises(TypeError):
            Bits('1') | 1

    def test_and_other_type_not_implemented(self):
        with pytest.raises(TypeError):
            Bits('1') & 1
