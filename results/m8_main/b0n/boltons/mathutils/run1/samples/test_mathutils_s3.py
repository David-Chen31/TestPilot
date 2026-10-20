import math

import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

def test_clamp_within_range():
    assert clamp(1.0, 0, 5) == 1.0
    assert clamp(3, 0, 5) == 3


def test_clamp_below_lower():
    result = clamp(-1.0, 0, 5)
    assert result == 0
    assert isinstance(result, int)


def test_clamp_above_upper():
    result = clamp(101.0, 0, 5)
    assert result == 5
    assert isinstance(result, int)


def test_clamp_only_upper():
    assert clamp(123, upper=5) == 5
    assert clamp(-123, upper=5) == -123


def test_clamp_only_lower():
    assert clamp(-5, lower=0) == 0
    assert clamp(5, lower=0) == 5


def test_clamp_no_bounds():
    assert clamp(123) == 123
    assert clamp(-123) == -123
    assert clamp(1.5) == 1.5


def test_clamp_equal_bounds():
    assert clamp(5, 3, 3) == 3
    assert clamp(3, 3, 3) == 3


def test_clamp_preserves_type_when_inside():
    result = clamp(2.5, 0, 5)
    assert result == 2.5
    assert isinstance(result, float)


def test_clamp_upper_less_than_lower_raises():
    with pytest.raises(ValueError):
        clamp(1, 5, 0)


def test_clamp_infinite_bounds():
    assert clamp(10, float('-inf'), float('inf')) == 10
    assert clamp(10, 0, float('inf')) == 10
    assert clamp(-10, float('-inf'), 0) == -10


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

def test_ceil_basic_no_options():
    assert ceil(1.2) == 2
    assert ceil(2.0) == 2
    assert ceil(-1.2) == -1
    assert ceil(0) == 0


def test_ceil_no_options_matches_math():
    for value in [-3.7, -1.0, 0.0, 0.1, 4.5, 10.0]:
        assert ceil(value) == math.ceil(value)


def test_ceil_with_options_docstring_examples():
    valid = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(3.5, options=valid) == 4
    assert ceil(4, options=valid) == 4


def test_ceil_with_options_exact_match():
    valid = [1.5, 2.5, 4, 6]
    assert ceil(2.5, options=valid) == 2.5


def test_ceil_with_options_less_than_all():
    valid = [10, 20, 30]
    assert ceil(0, options=valid) == 10
    assert ceil(-5, options=valid) == 10


def test_ceil_with_options_unsorted():
    valid = [50, 10, 30, 20]
    assert ceil(15, options=valid) == 20


def test_ceil_with_options_no_match_raises():
    valid = [1, 2, 3]
    with pytest.raises(ValueError):
        ceil(5, options=valid)


def test_ceil_with_options_unsorted_no_match_raises():
    valid = [3, 1, 2]
    with pytest.raises(ValueError):
        ceil(10, options=valid)


def test_ceil_with_empty_options_raises():
    with pytest.raises(ValueError):
        ceil(1, options=[])


def test_ceil_with_options_accepts_generator():
    assert ceil(3, options=iter([1, 5, 9])) == 5


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

def test_floor_basic_no_options():
    assert floor(1.8) == 1
    assert floor(2.0) == 2
    assert floor(-1.2) == -2
    assert floor(0) == 0


def test_floor_no_options_matches_math():
    for value in [-3.7, -1.0, 0.0, 0.1, 4.5, 10.0]:
        assert floor(value) == math.floor(value)


def test_floor_with_options_docstring_examples():
    valid = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert floor(3.5, options=valid) == 2.5
    assert floor(2.5, options=valid) == 2.5


def test_floor_with_options_exact_match():
    valid = [1.5, 2.5, 4, 6]
    assert floor(4, options=valid) == 4


def test_floor_with_options_greater_than_all():
    valid = [10, 20, 30]
    assert floor(100, options=valid) == 30


def test_floor_with_options_unsorted():
    valid = [50, 10, 30, 20]
    assert floor(25, options=valid) == 20


def test_floor_with_options_no_match_raises():
    valid = [10, 20, 30]
    with pytest.raises(ValueError):
        floor(5, options=valid)


def test_floor_with_empty_options_raises():
    with pytest.raises(ValueError):
        floor(1, options=[])


def test_floor_with_options_accepts_generator():
    assert floor(7, options=iter([1, 5, 9])) == 5


# ---------------------------------------------------------------------------
# Bits: construction
# ---------------------------------------------------------------------------

def test_bits_default_construction():
    b = Bits()
    assert b.val == 0
    assert b.len == 1
    assert b.as_bin() == '0'


def test_bits_from_int_auto_length():
    b = Bits(5)
    assert b.val == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_from_int_with_length():
    b = Bits(5, 8)
    assert b.val == 5
    assert b.len == 8
    assert b.as_bin() == '00000101'


def test_bits_from_int_zero():
    b = Bits(0)
    assert b.val == 0
    assert b.len == 1
    assert b.as_bin() == '0'


def test_bits_from_int_negative_raises():
    with pytest.raises(ValueError):
        Bits(-1)


def test_bits_from_int_too_large_for_length_raises():
    with pytest.raises(ValueError):
        Bits(8, 3)


def test_bits_from_int_exact_fit():
    b = Bits(7, 3)
    assert b.as_bin() == '111'


def test_bits_from_list():
    b = Bits([True, False, True])
    assert b.val == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_from_list_all_false():
    b = Bits([False, False, False])
    assert b.val == 0
    assert b.len == 3
    assert b.as_bin() == '000'


def test_bits_from_bin_string():
    b = Bits('101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_bin_string_leading_zeros():
    b = Bits('00101')
    assert b.val == 5
    assert b.len == 5
    assert b.as_bin() == '00101'


def test_bits_from_empty_string():
    b = Bits('')
    assert b.val == 0
    assert b.len == 0
    assert b.as_bin() == ''


def test_bits_from_hex_string():
    b = Bits('0x0F')
    assert b.val == 15
    assert b.len == 8
    assert b.as_bin() == '00001111'


def test_bits_from_hex_string_short():
    b = Bits('0xF')
    assert b.val == 15
    assert b.len == 4


def test_bits_from_bytes():
    b = Bits(b'101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_bad_type_raises():
    with pytest.raises(TypeError):
        Bits(1.5)


def test_bits_from_dict_raises():
    with pytest.raises(TypeError):
        Bits({})


def test_bits_from_none_raises():
    with pytest.raises(TypeError):
        Bits(None)


# ---------------------------------------------------------------------------
# Bits: indexing
# ---------------------------------------------------------------------------

def test_bits_getitem_int():
    b = Bits('101')
    assert b[0] is True
    assert b[1] is False
    assert b[2] is True


def test_bits_getitem_negative_index():
    b = Bits('101')
    assert b[-1] is True
    assert b[-2] is False
    assert b[-3] is True


def test_bits_getitem_out_of_range_raises():
    b = Bits('101')
    with pytest.raises(IndexError):
        b[3]
    with pytest.raises(IndexError):
        b[-4]


def test_bits_getitem_bad_type_raises():
    b = Bits('101')
    with pytest.raises(TypeError):
        b['a']


def test_bits_getitem_slice():
    b = Bits('1011')
    sliced = b[1:3]
    assert isinstance(sliced, Bits)
    assert sliced.as_bin() == '01'


def test_bits_getitem_slice_all():
    b = Bits('1011')
    assert b[:] == Bits('1011')


def test_bits_getitem_slice_with_step():
    b = Bits('1010')
    assert b[::2] == Bits('11')


# ---------------------------------------------------------------------------
# Bits: length, equality, hash
# ---------------------------------------------------------------------------

def test_bits_len():
    assert len(Bits('101')) == 3
    assert len(Bits(0)) == 1
    assert len(Bits('')) == 0


def test_bits_len_with_explicit_length():
    assert len(Bits(5, 8)) == 8


def test_bits_equality():
    assert Bits('101') == Bits('101')
    assert Bits(5) == Bits(5, 3)
    assert not (Bits('101') == Bits('1010'))


def test_bits_equality_different_lengths():
    assert Bits('01') != Bits('1')
    assert Bits(1, 2) != Bits(1, 3)


def test_bits_equality_other_type():
    b = Bits('101')
    assert b != '101'
    assert b != 5


def test_bits_hash():
    assert hash(Bits(5)) == hash(5)
    assert hash(Bits('101')) == hash(Bits(5, 3))


def test_bits_usable_in_set():
    s = {Bits('101'), Bits('101'), Bits('010')}
    assert len(s) == 2


# ---------------------------------------------------------------------------
# Bits: bitwise operators
# ---------------------------------------------------------------------------

def test_bits_or():
    a = Bits('100')
    b = Bits('011')
    result = a | b
    assert isinstance(result, Bits)
    assert result.val == 7
    assert result.len == 3


def test_bits_or_different_lengths():
    a = Bits(1, 2)   # '01'
    b = Bits(0, 5)   # '00000'
    result = a | b
    assert result.val == 1
    assert result.len == 5


def test_bits_or_other_type():
    with pytest.raises(TypeError):
        Bits(1) | 1


def test_bits_and():
    a = Bits('110')
    b = Bits('101')
    result = a & b
    assert isinstance(result, Bits)
    assert result.val == 4
    assert result.len == 3


def test_bits_and_different_lengths():
    a = Bits(3, 2)   # '11'
    b = Bits(6, 4)   # '0110'
    result = a & b
    assert result.val == 2
    assert result.len == 4


def test_bits_and_other_type():
    with pytest.raises(TypeError):
        Bits(1) & 1


def test_bits_lshift():
    b = Bits('1')
    result = b << 2
    assert result.val == 4
    assert result.len == 3
    assert result.as_bin() == '100'


def test_bits_rshift():
    b = Bits('100')
    result = b >> 2
    assert result.val == 1
    assert result.len == 1
    assert result.as_bin() == '1'


# ---------------------------------------------------------------------------
# Bits: representations
# ---------------------------------------------------------------------------

def test_bits_as_list():
    b = Bits('101')
    assert b.as_list() == [True, False, True]


def test_bits_as_list_empty():
    assert Bits('').as_list() == []


def test_bits_as_bin():
    assert Bits(5, 4).as_bin() == '0101'
    assert Bits(5).as_bin() == '101'


def test_bits_as_bin_empty():
    assert Bits('').as_bin() == ''


def test_bits_as_hex():
    assert Bits('00001111').as_hex() == '0F'
    assert Bits('1111').as_hex() == '0F'


def test_bits_as_hex_padding():
    # 9 bits -> 2 bytes -> 4 hex chars
    b = Bits('1' * 9)
    assert b.as_hex() == '01FF'


def test_bits_as_hex_empty():
    assert Bits('').as_hex() == ''


def test_bits_as_int():
    assert Bits('101').as_int() == 5
    assert Bits('').as_int() == 0


def test_bits_as_bytes():
    b = Bits('00000001')
    assert b.as_bytes() == b'\x01'


def test_bits_as_bytes_multi():
    b = Bits('0000000100000010')
    assert b.as_bytes() == b'\x01\x02'


# ---------------------------------------------------------------------------
# Bits: classmethods
# ---------------------------------------------------------------------------

def test_bits_from_list_classmethod():
    b = Bits.from_list([True, False, True])
    assert b == Bits('101')


def test_bits_from_bin_classmethod():
    b = Bits.from_bin('101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_hex_classmethod():
    b = Bits.from_hex('0F')
    assert b.val == 15
    assert b.len == 8


def test_bits_from_hex_with_prefix():
    b = Bits.from_hex('0x0F')
    assert b.val == 15
    assert b.len == 8


def test_bits_from_hex_bytes():
    b = Bits.from_hex(b'0F')
    assert b.val == 15
    assert b.len == 8


def test_bits_from_hex_empty():
    b = Bits.from_hex('')
    assert b.val == 0
    assert b.len == 0


def test_bits_from_hex_empty_bytes():
    b = Bits.from_hex(b'')
    assert b.val == 0
    assert b.len == 0


def test_bits_from_int_classmethod():
    b = Bits.from_int(5)
    assert b.val == 5
    assert b.len == 3


def test_bits_from_int_classmethod_with_len():
    b = Bits.from_int(5, 8)
    assert b.val == 5
    assert b.len == 8


def test_bits_from_bytes_classmethod():
    b = Bits.from_bytes(b'\x01\x02')
    assert b.val == 0x0102
    assert b.as_bytes() == b'\x01\x02'


# ---------------------------------------------------------------------------
# Bits: repr
# ---------------------------------------------------------------------------

def test_bits_repr():
    b = Bits('101')
    assert repr(b) == "Bits('101')"


def test_bits_repr_padded():
    b = Bits(5, 8)
    assert repr(b) == "Bits('00000101')"


# ---------------------------------------------------------------------------
# Bits: immutability / slots
# ---------------------------------------------------------------------------

def test_bits_has_no_dict():
    b = Bits('101')
    assert not hasattr(b, '__dict__')


def test_bits_roundtrip_int():
    for value in [0, 1, 2, 5, 255, 1024]:
        b = Bits(value)
        assert Bits.from_int(b.as_int()) == b


def test_bits_roundtrip_bytes():
    data = b'\xde\xad'
    b = Bits.from_bytes(data)
    assert b.as_bytes() == data
