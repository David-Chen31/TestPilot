import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

def test_clamp_within_range():
    assert clamp(1.0, 0, 5) == 1.0
    assert clamp(3, 0, 5) == 3
    assert clamp(2.5, 0.0, 10.0) == 2.5


def test_clamp_below_lower():
    assert clamp(-1.0, 0, 5) == 0
    assert clamp(-100, lower=-10) == -10


def test_clamp_above_upper():
    assert clamp(101.0, 0, 5) == 5
    assert clamp(123, upper=5) == 5


def test_clamp_defaults():
    assert clamp(123456789) == 123456789
    assert clamp(-123456789) == -123456789


def test_clamp_equal_bounds():
    assert clamp(10, 5, 5) == 5
    assert clamp(1, 5, 5) == 5


def test_clamp_upper_less_than_lower():
    with pytest.raises(ValueError):
        clamp(5, 10, 0)


def test_clamp_mixed_numeric_types():
    assert clamp(1.5, 0, 2) == 1.5
    assert isinstance(clamp(-1.5, 0, 2), int)


def test_clamp_upper_less_than_lower_message():
    with pytest.raises(ValueError) as excinfo:
        clamp(5, 10, 0)
    assert 'upper bound' in str(excinfo.value)
    assert 'lower bound' in str(excinfo.value)


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

def test_ceil_plain():
    assert ceil(1.1) == 2
    assert ceil(2.0) == 2
    assert ceil(-1.1) == -1
    assert ceil(-2.0) == -2
    assert ceil(5) == 5


def test_ceil_options():
    VALID = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(3.5, options=VALID) == 4
    assert ceil(4, options=VALID) == 4
    assert ceil(0, options=VALID) == 1.5
    assert ceil(1.5, options=VALID) == 1.5
    assert ceil(50, options=VALID) == 50


def test_ceil_options_unsorted_input():
    assert ceil(3.5, options=[10, 1.5, 4]) == 4


def test_ceil_options_generator():
    assert ceil(3.5, options=(x for x in [1.5, 2.5, 4, 6])) == 4


def test_ceil_options_no_match():
    with pytest.raises(ValueError):
        ceil(100, options=[1, 2, 3])


def test_ceil_options_no_match_message():
    with pytest.raises(ValueError) as excinfo:
        ceil(100, options=[1, 2, 3])
    assert 'no ceil options greater than or equal to' in str(excinfo.value)


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

def test_floor_plain():
    assert floor(1.9) == 1
    assert floor(2.0) == 2
    assert floor(-1.1) == -2
    assert floor(-2.0) == -2
    assert floor(5) == 5


def test_floor_options():
    VALID = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert floor(3.5, options=VALID) == 2.5
    assert floor(2.5, options=VALID) == 2.5
    assert floor(4, options=VALID) == 4
    assert floor(51, options=VALID) == 50


def test_floor_options_unsorted_input():
    assert floor(3.5, options=[10, 1.5, 4]) == 1.5


def test_floor_options_generator():
    assert floor(3.5, options=(x for x in [1.5, 2.5, 4, 6])) == 2.5


def test_floor_options_no_match():
    with pytest.raises(ValueError):
        floor(0, options=[1, 2, 3])


def test_floor_options_no_match_message():
    with pytest.raises(ValueError) as excinfo:
        floor(0, options=[1, 2, 3])
    assert 'no floor options less than or equal to' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Bits: construction
# ---------------------------------------------------------------------------

def test_bits_default():
    b = Bits()
    assert b.as_int() == 0
    assert b.len == 1
    assert b.as_bin() == '0'
    assert len(b) == 1


def test_bits_from_int():
    b = Bits(5)
    assert b.val == 5
    assert b.len == 3
    assert b.as_bin() == '101'
    assert b.as_int() == 5


def test_bits_from_int_with_len():
    b = Bits(5, 8)
    assert b.val == 5
    assert b.len == 8
    assert b.as_bin() == '00000101'


def test_bits_from_list():
    b = Bits([True, False, True])
    assert b.as_bin() == '101'
    assert b.as_int() == 5
    assert b.as_list() == [True, False, True]


def test_bits_from_bin_string():
    b = Bits('1010')
    assert b.as_int() == 10
    assert b.len == 4


def test_bits_from_empty_bin_string():
    b = Bits('')
    assert b.as_int() == 0
    assert b.len == 0
    assert b.as_bin() == ''
    assert b.as_hex() == ''


def test_bits_from_hex_string():
    b = Bits('0xFF')
    assert b.as_int() == 255
    assert b.len == 8
    assert b.as_hex() == 'FF'


def test_bits_from_bytes():
    b = Bits(b'101')
    assert b.as_int() == 5


def test_bits_from_bytes_with_len():
    # bytes path decodes to str; when len_ is provided, that length is used
    b = Bits(b'101', 5)
    assert b.as_int() == 5
    assert b.len == 5
    assert b.as_bin() == '00101'


def test_bits_from_bin_string_with_explicit_len():
    # when len_ is provided, the "0x" prefix-based length is skipped
    # and the string is parsed as binary
    b = Bits('101', 6)
    assert b.as_int() == 5
    assert b.len == 6
    assert b.as_bin() == '000101'


def test_bits_from_hex_string_with_explicit_len():
    # explicit len_ provided; "0x" is still detected and parsed as hex
    b = Bits('0xF', 8)
    assert b.as_int() == 15
    assert b.len == 8
    assert b.as_hex() == '0F'


def test_bits_negative_value():
    with pytest.raises(ValueError):
        Bits(-1)


def test_bits_negative_value_message():
    with pytest.raises(ValueError) as excinfo:
        Bits(-1)
    assert 'cannot represent negative values' in str(excinfo.value)


def test_bits_value_too_big():
    with pytest.raises(ValueError):
        Bits(8, 3)


def test_bits_value_too_big_message():
    with pytest.raises(ValueError) as excinfo:
        Bits(8, 3)
    msg = str(excinfo.value)
    assert '8' in msg
    assert '3' in msg


def test_bits_bad_type():
    with pytest.raises(TypeError):
        Bits(object())


def test_bits_bad_int_float():
    with pytest.raises(TypeError):
        Bits(1.5)


# ---------------------------------------------------------------------------
# Bits: indexing / len
# ---------------------------------------------------------------------------

def test_bits_getitem():
    b = Bits('1011')
    assert b[0] is True
    assert b[1] is False
    assert b[2] is True
    assert b[3] is True
    assert b[-1] is True
    assert b[-2] is True


def test_bits_getitem_out_of_range():
    b = Bits('1011')
    with pytest.raises(IndexError):
        b[4]
    with pytest.raises(IndexError):
        b[-5]


def test_bits_getitem_bad_type():
    b = Bits('1011')
    with pytest.raises(TypeError):
        b['0']


def test_bits_getitem_slice():
    b = Bits('10110')
    s = b[1:4]
    assert isinstance(s, Bits)
    assert s.as_bin() == '011'


# ---------------------------------------------------------------------------
# Bits: equality / hashing
# ---------------------------------------------------------------------------

def test_bits_equality():
    assert Bits('101') == Bits('101')
    assert not (Bits('101') == Bits('010'))


def test_bits_equality_different_len():
    assert not (Bits(5, 3) == Bits(5, 4))


def test_bits_equality_other_type():
    assert (Bits('101') == '101') is False


def test_bits_hash():
    assert hash(Bits(5)) == hash(5)
    assert hash(Bits('101')) == hash(Bits(5))


# ---------------------------------------------------------------------------
# Bits: bitwise ops
# ---------------------------------------------------------------------------

def test_bits_or():
    a = Bits('1010')
    b = Bits('0110')
    r = a | b
    assert r.as_bin() == '1110'
    assert r.len == 4


def test_bits_or_different_len():
    a = Bits('1')
    b = Bits('1000')
    r = a | b
    assert r.len == 4
    assert r.as_int() == 0b1001


def test_bits_and():
    a = Bits('1010')
    b = Bits('0110')
    r = a & b
    assert r.as_bin() == '0010'


def test_bits_and_different_len():
    a = Bits('1')
    b = Bits('1000')
    r = a & b
    assert r.len == 4
    assert r.as_int() == 0


def test_bits_or_other_type():
    with pytest.raises(TypeError):
        Bits('101') | 1


def test_bits_and_other_type():
    with pytest.raises(TypeError):
        Bits('101') & 1


def test_bits_lshift():
    b = Bits('1')
    r = b << 3
    assert r.as_bin() == '1000'
    assert r.len == 4


def test_bits_rshift():
    b = Bits('1000')
    r = b >> 2
    assert r.as_int() == 0b10
    assert r.len == 2


# ---------------------------------------------------------------------------
# Bits: conversions
# ---------------------------------------------------------------------------

def test_bits_as_list():
    assert Bits(5).as_list() == [True, False, True]


def test_bits_as_bin_padding():
    assert Bits(1, 4).as_bin() == '0001'


def test_bits_as_hex():
    assert Bits(255).as_hex() == 'FF'
    assert Bits(0, 8).as_hex() == '00'
    assert Bits(1, 4).as_hex() == '01'


def test_bits_as_hex_padding():
    # 9 bits -> 2 bytes = 4 hex chars
    assert Bits(1, 9).as_hex() == '0001'


def test_bits_as_bytes():
    assert Bits(0xAB, 8).as_bytes() == b'\xab'


def test_bits_repr():
    assert repr(Bits('101')) == "Bits('101')"


# ---------------------------------------------------------------------------
# Bits: classmethods
# ---------------------------------------------------------------------------

def test_bits_from_list():
    b = Bits.from_list([True, False, True])
    assert b.as_bin() == '101'


def test_bits_from_bin():
    b = Bits.from_bin('101')
    assert b.as_int() == 5


def test_bits_from_hex():
    b = Bits.from_hex('FF')
    assert b.as_int() == 255
    assert b.len == 8


def test_bits_from_hex_with_prefix():
    b = Bits.from_hex('0xFF')
    assert b.as_int() == 255


def test_bits_from_hex_bytes():
    b = Bits.from_hex(b'FF')
    assert b.as_int() == 255


def test_bits_from_hex_empty():
    b = Bits.from_hex('')
    assert b.as_int() == 0
    assert b.len == 0


def test_bits_from_int_cm():
    b = Bits.from_int(5)
    assert b.as_int() == 5
    b2 = Bits.from_int(5, 8)
    assert b2.as_bin() == '00000101'


def test_bits_from_bytes():
    b = Bits.from_bytes(b'\xab')
    assert b.as_int() == 0xAB
    assert b.as_bytes() == b'\xab'


def test_bits_roundtrip_bin():
    for val in [0, 1, 2, 5, 255]:
        b = Bits(val)
        assert Bits.from_bin(b.as_bin()).val == val


def test_bits_roundtrip_hex():
    for val in [0, 1, 15, 16, 255]:
        b = Bits(val, 8) if val > 0 else Bits(0, 8)
        assert Bits.from_hex(b.as_hex()).val == val
