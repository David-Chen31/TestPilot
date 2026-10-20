import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------- clamp

def test_clamp_no_bounds():
    assert clamp(5) == 5
    assert clamp(-1.5) == -1.5


def test_clamp_within_range():
    assert clamp(1.0, 0, 5) == 1.0
    assert clamp(3, 0, 5) == 3


def test_clamp_below_lower():
    assert clamp(-1.0, 0, 5) == 0
    assert clamp(-100, lower=0) == 0


def test_clamp_above_upper():
    assert clamp(101.0, 0, 5) == 5
    assert clamp(123, upper=5) == 5


def test_clamp_only_lower():
    assert clamp(10, lower=0) == 10
    assert clamp(-10, lower=0) == 0


def test_clamp_only_upper():
    assert clamp(10, upper=5) == 5
    assert clamp(-10, upper=5) == -10


def test_clamp_equal_bounds():
    assert clamp(5, 3, 3) == 3
    assert clamp(1, 3, 3) == 3


def test_clamp_invalid_range():
    with pytest.raises(ValueError):
        clamp(1, 5, 0)


def test_clamp_mixed_types():
    assert clamp(2.5, 1, 3) == 2.5
    assert clamp(0, 1.0, 3.0) == 1.0


# ---------------------------------------------------------------- ceil

def test_ceil_basic():
    assert ceil(3.2) == 4
    assert ceil(4.0) == 4
    assert ceil(-3.2) == -3
    assert ceil(0) == 0


def test_ceil_options():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(3.5, options=VALID_CABLE_CSA) == 4
    assert ceil(4, options=VALID_CABLE_CSA) == 4
    assert ceil(1.0, options=VALID_CABLE_CSA) == 1.5
    assert ceil(50, options=VALID_CABLE_CSA) == 50


def test_ceil_options_unsorted():
    assert ceil(3.5, options=[10, 1.5, 4, 2.5]) == 4


def test_ceil_options_no_match():
    with pytest.raises(ValueError):
        ceil(100, options=[1, 2, 3])


def test_ceil_options_empty():
    with pytest.raises(ValueError):
        ceil(1, options=[])


def test_ceil_options_exact_match():
    assert ceil(2.5, options=[1.5, 2.5, 4]) == 2.5


# ---------------------------------------------------------------- floor

def test_floor_basic():
    assert floor(3.7) == 3
    assert floor(4.0) == 4
    assert floor(-3.2) == -4
    assert floor(0) == 0


def test_floor_options():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert floor(3.5, options=VALID_CABLE_CSA) == 2.5
    assert floor(2.5, options=VALID_CABLE_CSA) == 2.5
    assert floor(50, options=VALID_CABLE_CSA) == 50
    assert floor(1.5, options=VALID_CABLE_CSA) == 1.5


def test_floor_options_unsorted():
    assert floor(3.5, options=[10, 1.5, 4, 2.5]) == 2.5


def test_floor_options_no_match():
    with pytest.raises(ValueError):
        floor(1, options=[2, 3, 4])


def test_floor_options_empty():
    with pytest.raises(ValueError):
        floor(1, options=[])


# ---------------------------------------------------------------- Bits

def test_bits_default():
    b = Bits()
    assert b.as_int() == 0
    assert b.len == 1
    assert b.as_bin() == '0'


def test_bits_from_int():
    b = Bits(5)
    assert b.as_int() == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_int_with_len():
    b = Bits(5, 8)
    assert b.as_int() == 5
    assert b.len == 8
    assert b.as_bin() == '00000101'


def test_bits_int_too_big():
    with pytest.raises(ValueError):
        Bits(8, 3)


def test_bits_negative():
    with pytest.raises(ValueError):
        Bits(-1)


def test_bits_from_bin_string():
    b = Bits('101')
    assert b.as_int() == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_from_bin_leading_zeros():
    b = Bits('000101')
    assert b.len == 6
    assert b.as_int() == 5
    assert b.as_bin() == '000101'


def test_bits_empty_string():
    b = Bits('')
    assert b.as_int() == 0
    assert b.len == 0
    assert b.as_bin() == ''
    assert b.as_list() == []


def test_bits_from_hex_string():
    b = Bits('0xFF')
    assert b.as_int() == 255
    assert b.len == 8


def test_bits_from_bytes():
    b = Bits(b'101')
    assert b.as_int() == 5
    assert b.len == 3


def test_bits_from_list():
    b = Bits([True, False, True])
    assert b.as_int() == 5
    assert b.len == 3


def test_bits_from_invalid_list():
    b = Bits([1, 0, 1])
    assert b.as_int() == 5


def test_bits_bad_type():
    with pytest.raises(TypeError):
        Bits(1.5)


def test_bits_getitem():
    b = Bits('1011')
    assert b[0] is True
    assert b[1] is False
    assert b[2] is True
    assert b[3] is True


def test_bits_getitem_negative():
    b = Bits('1011')
    assert b[-1] is True
    assert b[-2] is True
    assert b[-3] is False
    assert b[-4] is True


def test_bits_getitem_out_of_range():
    b = Bits('1011')
    with pytest.raises(IndexError):
        b[4]
    with pytest.raises(IndexError):
        b[-5]


def test_bits_getitem_bad_type():
    b = Bits('1011')
    with pytest.raises(TypeError):
        b['x']


def test_bits_getitem_slice():
    b = Bits('1011')
    s = b[1:3]
    assert isinstance(s, Bits)
    assert s.as_bin() == '01'


def test_bits_len():
    assert len(Bits('1011')) == 4
    assert len(Bits()) == 1
    assert len(Bits('')) == 0


def test_bits_eq():
    assert Bits('101') == Bits('101')
    assert Bits(5, 4) != Bits(5, 3)
    assert Bits(5) == Bits(5)


def test_bits_eq_other_type():
    assert Bits('101') != 5
    assert Bits('101') != '101'


def test_bits_hash():
    assert hash(Bits(5)) == hash(5)
    assert hash(Bits('101')) == hash(5)


def test_bits_or():
    a = Bits('1010')
    b = Bits('0110')
    c = a | b
    assert c.as_bin() == '1110'


def test_bits_or_diff_len():
    a = Bits('10')        # len 2
    b = Bits('0011')      # len 4
    c = a | b
    assert c.len == 4
    assert c.as_int() == 2 | 3


def test_bits_and():
    a = Bits('1010')
    b = Bits('0110')
    c = a & b
    assert c.as_bin() == '0010'


def test_bits_and_diff_len():
    a = Bits('10')
    b = Bits('0011')
    c = a & b
    assert c.len == 4


def test_bits_or_other_type():
    a = Bits('10')
    with pytest.raises(TypeError):
        a | 5


def test_bits_and_other_type():
    a = Bits('10')
    with pytest.raises(TypeError):
        a & 5


def test_bits_lshift():
    b = Bits('101')
    c = b << 2
    assert c.len == 5
    assert c.as_int() == 5 << 2


def test_bits_rshift():
    b = Bits('10100')
    c = b >> 2
    assert c.len == 3
    assert c.as_int() == (20 >> 2)


def test_bits_as_list():
    assert Bits('1011').as_list() == [True, False, True, True]
    assert Bits('').as_list() == []


def test_bits_as_bin():
    assert Bits(5, 3).as_bin() == '101'
    assert Bits(5, 8).as_bin() == '00000101'
    assert Bits('').as_bin() == ''


def test_bits_as_hex():
    assert Bits('11111111').as_hex() == 'FF'
    assert Bits('1010').as_hex() == 'A'
    assert Bits('').as_hex() == ''


def test_bits_as_hex_padding():
    # 9 bits -> 2 bytes
    b = Bits('100000000')
    assert b.as_hex() == '0100'


def test_bits_as_bytes():
    assert Bits('11111111').as_bytes() == b'\xff'
    assert Bits('01000001').as_bytes() == b'A'


def test_bits_as_int():
    assert Bits('101').as_int() == 5


def test_bits_from_list_classmethod():
    b = Bits.from_list([True, False, True])
    assert b.as_int() == 5
    assert b.len == 3


def test_bits_from_bin_classmethod():
    b = Bits.from_bin('101')
    assert b.as_int() == 5


def test_bits_from_hex_classmethod():
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


def test_bits_from_int_classmethod():
    b = Bits.from_int(5)
    assert b.as_int() == 5
    assert b.len == 3


def test_bits_from_int_with_len():
    b = Bits.from_int(5, 8)
    assert b.len == 8


def test_bits_from_bytes_classmethod():
    b = Bits.from_bytes(b'\xff')
    assert b.as_int() == 255
    assert b.len == 8


def test_bits_repr():
    b = Bits('101')
    assert repr(b) == "Bits('101')"


def test_bits_slots():
    b = Bits('101')
    with pytest.raises(AttributeError):
        b.new_attr = 1
