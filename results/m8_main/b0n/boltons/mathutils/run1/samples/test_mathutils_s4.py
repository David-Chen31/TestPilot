import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

def test_clamp_basic():
    assert clamp(1.0, 0, 5) == 1.0
    assert clamp(-1.0, 0, 5) == 0
    assert clamp(101.0, 0, 5) == 5
    assert clamp(123, upper=5) == 5
    assert clamp(-123, lower=0) == 0


def test_clamp_defaults():
    assert clamp(42) == 42
    assert clamp(-42) == -42
    assert clamp(0) == 0


def test_clamp_inclusive_bounds():
    assert clamp(0, 0, 5) == 0
    assert clamp(5, 0, 5) == 5


def test_clamp_mixed_types():
    assert clamp(2.5, 0, 5) == 2.5
    assert clamp(2, 1.5, 5.5) == 2


def test_clamp_equal_bounds():
    assert clamp(3, 3, 3) == 3
    assert clamp(1, 3, 3) == 3
    assert clamp(10, 3, 3) == 3


def test_clamp_invalid_bounds():
    with pytest.raises(ValueError):
        clamp(1, 5, 0)
    with pytest.raises(ValueError):
        clamp(1, upper=-1, lower=0)


def test_clamp_return_type_follows_min_max():
    # When x is between bounds, the original object is returned.
    x = 3
    assert clamp(x, 0, 5) is x


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

def test_ceil_without_options():
    assert ceil(3.2) == 4
    assert ceil(3.0) == 3
    assert ceil(-3.2) == -3
    assert ceil(-3.0) == -3
    assert ceil(0) == 0


def test_ceil_with_options():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(3.5, options=VALID_CABLE_CSA) == 4
    assert ceil(4, options=VALID_CABLE_CSA) == 4
    assert ceil(2.5, options=VALID_CABLE_CSA) == 2.5
    assert ceil(0, options=VALID_CABLE_CSA) == 1.5
    assert ceil(1.5, options=VALID_CABLE_CSA) == 1.5


def test_ceil_options_unsorted_input():
    assert ceil(3.5, options=[50, 1.5, 10, 2.5, 4]) == 4


def test_ceil_options_does_not_mutate_input():
    options = [50, 1.5, 10, 2.5]
    original = list(options)
    ceil(3.5, options=options)
    assert options == original


def test_ceil_options_exact_upper_bound():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(50, options=VALID_CABLE_CSA) == 50


def test_ceil_options_out_of_range():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    with pytest.raises(ValueError):
        ceil(51, options=VALID_CABLE_CSA)


def test_ceil_options_empty():
    with pytest.raises(ValueError):
        ceil(1, options=[])


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

def test_floor_without_options():
    assert floor(3.7) == 3
    assert floor(3.0) == 3
    assert floor(-3.2) == -4
    assert floor(-3.0) == -3
    assert floor(0) == 0


def test_floor_with_options():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert floor(3.5, options=VALID_CABLE_CSA) == 2.5
    assert floor(2.5, options=VALID_CABLE_CSA) == 2.5
    assert floor(4, options=VALID_CABLE_CSA) == 4
    assert floor(50, options=VALID_CABLE_CSA) == 50
    assert floor(49, options=VALID_CABLE_CSA) == 35


def test_floor_options_unsorted_input():
    assert floor(3.5, options=[50, 1.5, 10, 2.5]) == 2.5


def test_floor_options_does_not_mutate_input():
    options = [50, 1.5, 10, 2.5]
    original = list(options)
    floor(3.5, options=options)
    assert options == original


def test_floor_options_out_of_range():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    with pytest.raises(ValueError):
        floor(1.0, options=VALID_CABLE_CSA)


def test_floor_options_empty():
    with pytest.raises(ValueError):
        floor(1, options=[])


# ---------------------------------------------------------------------------
# Bits - construction
# ---------------------------------------------------------------------------

def test_bits_default():
    b = Bits()
    assert b.as_int() == 0
    assert b.len == 1
    assert len(b) == 1
    assert b.as_bin() == '0'


def test_bits_from_int():
    b = Bits(5)
    assert b.as_int() == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_from_int_with_len():
    b = Bits(5, len_=8)
    assert b.as_int() == 5
    assert b.len == 8
    assert b.as_bin() == '00000101'


def test_bits_from_int_zero_explicit_len():
    b = Bits(0, len_=4)
    assert b.as_int() == 0
    assert b.len == 4
    assert b.as_bin() == '0000'


def test_bits_from_list():
    b = Bits([True, False, True, True])
    assert b.as_int() == 11
    assert b.len == 4
    assert b.as_bin() == '1011'
    assert b.as_list() == [True, False, True, True]


def test_bits_from_bin_string():
    b = Bits('1011')
    assert b.as_int() == 11
    assert b.len == 4


def test_bits_from_empty_string():
    b = Bits('')
    assert b.as_int() == 0
    assert b.len == 0


def test_bits_from_hex_string():
    b = Bits('0xA5')
    assert b.as_int() == 0xA5
    assert b.len == 8
    assert b.as_hex() == 'A5'


def test_bits_from_hex_string_short():
    b = Bits('0xF')
    assert b.as_int() == 15
    assert b.len == 4


def test_bits_from_bytes():
    b = Bits(b'1010')
    assert b.as_int() == 10
    assert b.len == 4


def test_bits_negative_value():
    with pytest.raises(ValueError):
        Bits(-1)


def test_bits_value_too_large_for_len():
    with pytest.raises(ValueError):
        Bits(8, len_=3)


def test_bits_bad_type():
    with pytest.raises(TypeError):
        Bits(1.5)


# ---------------------------------------------------------------------------
# Bits - access
# ---------------------------------------------------------------------------

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


def test_bits_getitem_wrong_type():
    b = Bits('1011')
    with pytest.raises(TypeError):
        b['a']


def test_bits_getitem_slice():
    b = Bits('101101')
    s = b[1:4]
    assert isinstance(s, Bits)
    assert s.as_bin() == '011'


def test_bits_len():
    assert len(Bits('1011')) == 4
    assert len(Bits('')) == 0
    assert len(Bits(0, len_=10)) == 10


# ---------------------------------------------------------------------------
# Bits - equality
# ---------------------------------------------------------------------------

def test_bits_eq():
    assert Bits('101') == Bits('101')
    assert Bits(5) == Bits(5, len_=3)
    assert not (Bits('101') == Bits('1010'))


def test_bits_eq_other_type():
    assert Bits('101').__eq__(5) is NotImplemented
    assert Bits('101') != 5


def test_bits_hash():
    assert hash(Bits(5)) == hash(5)
    assert hash(Bits(5)) == hash(Bits(5, len_=10))


# ---------------------------------------------------------------------------
# Bits - bitwise ops
# ---------------------------------------------------------------------------

def test_bits_or():
    a = Bits('1010')
    b = Bits('0101')
    r = a | b
    assert isinstance(r, Bits)
    assert r.as_int() == 0b1111
    assert r.len == 4


def test_bits_or_different_lengths():
    a = Bits('10')
    b = Bits('0001', len_=4) if False else Bits(1, len_=4)  # 0001
    r = a | b
    assert r.as_int() == 0b1011
    assert r.len == 4


def test_bits_and():
    a = Bits('1010')
    b = Bits('0110')
    r = a & b
    assert isinstance(r, Bits)
    assert r.as_int() == 0b0010
    assert r.len == 4


def test_bits_and_different_lengths():
    a = Bits('1010')
    b = Bits('10', len_=2)
    r = a & b
    assert r.as_int() == 0b10
    assert r.len == 4


def test_bits_or_other_type():
    assert Bits('1').__or__(1) is NotImplemented
    assert Bits('1').__and__(1) is NotImplemented


def test_bits_lshift():
    b = Bits('101').as_int()  # 5
    a = Bits(5, len_=3)
    r = a << 2
    assert isinstance(r, Bits)
    assert r.as_int() == 0b10100
    assert r.len == 5


def test_bits_rshift():
    a = Bits(0b10100, len_=5)
    r = a >> 2
    assert isinstance(r, Bits)
    assert r.as_int() == 0b101
    assert r.len == 3


# ---------------------------------------------------------------------------
# Bits - conversions
# ---------------------------------------------------------------------------

def test_bits_as_list():
    assert Bits('1011').as_list() == [True, False, True, True]
    assert Bits('').as_list() == []


def test_bits_as_bin():
    assert Bits('1011').as_bin() == '1011'
    assert Bits(5, len_=6).as_bin() == '000101'
    assert Bits('').as_bin() == ''


def test_bits_as_hex():
    assert Bits('11111111').as_hex() == 'FF'
    assert Bits('10100101').as_hex() == 'A5'
    assert Bits('').as_hex() == ''


def test_bits_as_hex_padding():
    # 4 bits -> 1 byte
    assert Bits('1111').as_hex() == '0F'
    # 12 bits -> 2 bytes
    assert Bits(0xFFF, len_=12).as_hex() == '0FFF'


def test_bits_as_hex_single_digit():
    # len 4 -> 1 byte -> 2 hex chars
    b = Bits('1111')
    assert len(b.as_hex()) == 2


def test_bits_as_int():
    assert Bits('1011').as_int() == 11
    assert Bits('').as_int() == 0


def test_bits_as_bytes():
    b = Bits('10100101')  # 0xA5
    assert b.as_bytes() == b'\xa5'


def test_bits_as_bytes_padding():
    b = Bits('1111')  # 0x0F
    assert b.as_bytes() == b'\x0f'


# ---------------------------------------------------------------------------
# Bits - classmethods
# ---------------------------------------------------------------------------

def test_bits_from_list():
    b = Bits.from_list([True, False, True])
    assert b.as_bin() == '101'
    assert b.as_int() == 5


def test_bits_from_bin():
    b = Bits.from_bin('1101')
    assert b.as_int() == 13
    assert b.len == 4


def test_bits_from_hex():
    b = Bits.from_hex('A5')
    assert b.as_int() == 0xA5
    assert b.len == 8


def test_bits_from_hex_with_prefix():
    b = Bits.from_hex('0xA5')
    assert b.as_int() == 0xA5
    assert b.len == 8


def test_bits_from_hex_bytes():
    b = Bits.from_hex(b'A5')
    assert b.as_int() == 0xA5
    assert b.len == 8


def test_bits_from_hex_empty():
    b = Bits.from_hex('')
    assert b.as_int() == 0
    assert b.len == 0


def test_bits_from_int_classmethod():
    b = Bits.from_int(5)
    assert b.as_int() == 5
    assert b.len == 3
    b2 = Bits.from_int(5, len_=8)
    assert b2.len == 8


def test_bits_from_bytes():
    b = Bits.from_bytes(b'\xa5')
    assert b.as_int() == 0xA5
    assert b.len == 8


def test_bits_from_bytes_empty():
    b = Bits.from_bytes(b'')
    assert b.as_int() == 0
    assert b.len == 0


# ---------------------------------------------------------------------------
# Bits - repr
# ---------------------------------------------------------------------------

def test_bits_repr():
    b = Bits('1011')
    assert repr(b) == "Bits('1011')"


def test_bits_repr_empty():
    b = Bits('')
    assert repr(b) == "Bits('')"


def test_bits_slots_immutable_attrs():
    # attributes exist but there is no __dict__ due to slots
    b = Bits(3)
    assert b.val == 3
    assert b.len == 2
    with pytest.raises(AttributeError):
        b.__dict__
