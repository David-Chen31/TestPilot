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
    assert clamp(123, lower=200) == 200


def test_clamp_inside_range():
    assert clamp(3, 0, 5) == 3
    assert clamp(3.5, 0, 5) == 3.5


def test_clamp_defaults():
    assert clamp(42) == 42
    assert clamp(-1000.5) == -1000.5


def test_clamp_lower_equals_upper():
    assert clamp(10, 5, 5) == 5
    assert clamp(0, 5, 5) == 5


def test_clamp_invalid_bounds():
    with pytest.raises(ValueError):
        clamp(1, 5, 0)
    with pytest.raises(ValueError):
        clamp(1, upper=-1, lower=1)


def test_clamp_returns_bound_object_when_clamped():
    # when clamped, the returned value is the bound itself
    lower, upper = 0, 5
    assert clamp(-1, lower, upper) == lower
    assert clamp(10, lower, upper) == upper


def test_clamp_mixed_types():
    assert clamp(2.5, 0, 5) == 2.5
    assert clamp(2.5, 1, 2) == 2


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

def test_ceil_basic():
    assert ceil(1.2) == 2
    assert ceil(1.0) == 1
    assert ceil(-1.2) == -1
    assert ceil(0) == 0
    assert isinstance(ceil(1.2), int)


def test_ceil_options():
    opts = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(3.5, options=opts) == 4
    assert ceil(4, options=opts) == 4
    assert ceil(1.4, options=opts) == 1.5
    assert ceil(1.5, options=opts) == 1.5
    assert ceil(50, options=opts) == 50


def test_ceil_options_unsorted():
    opts = [50, 1.5, 4, 2.5, 35, 6, 10, 25]
    assert ceil(3.5, options=opts) == 4
    assert ceil(1.4, options=opts) == 1.5


def test_ceil_options_returns_exact_member():
    opts = [1.5, 4]
    # the exact value present should be returned unchanged
    assert ceil(4, options=opts) == 4
    assert ceil(1.5, options=opts) == 1.5


def test_ceil_options_no_result():
    opts = [1, 2, 3]
    with pytest.raises(ValueError):
        ceil(4, options=opts)
    with pytest.raises(ValueError):
        ceil(10, options=[])


def test_ceil_options_iterator():
    assert ceil(3.5, options=iter([1, 4, 6])) == 4


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

def test_floor_basic():
    assert floor(1.8) == 1
    assert floor(1.0) == 1
    assert floor(-1.2) == -2
    assert floor(0) == 0
    assert isinstance(floor(1.8), int)


def test_floor_options():
    opts = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert floor(3.5, options=opts) == 2.5
    assert floor(2.5, options=opts) == 2.5
    assert floor(100, options=opts) == 50
    assert floor(1.6, options=opts) == 1.5


def test_floor_options_unsorted():
    opts = [50, 1.5, 4, 2.5, 35, 6, 10, 25]
    assert floor(3.5, options=opts) == 2.5
    assert floor(100, options=opts) == 50


def test_floor_options_no_result():
    opts = [4, 6, 10]
    with pytest.raises(ValueError):
        floor(3, options=opts)
    with pytest.raises(ValueError):
        floor(1, options=[])


def test_floor_options_iterator():
    assert floor(3.5, options=iter([1, 4, 6])) == 1


# ---------------------------------------------------------------------------
# Bits: construction
# ---------------------------------------------------------------------------

def test_bits_default():
    b = Bits()
    assert b.val == 0
    assert b.len == 1  # len('0') == 1
    assert b.as_bin() == '0'


def test_bits_from_int():
    b = Bits(5)
    assert b.val == 5
    assert b.len == 3  # '101'
    assert b.as_bin() == '101'


def test_bits_from_int_with_len():
    b = Bits(5, 4)
    assert b.as_bin() == '0101'
    assert len(b) == 4


def test_bits_int_too_big_for_len():
    with pytest.raises(ValueError):
        Bits(5, 2)


def test_bits_zero_with_len():
    b = Bits(0, 4)
    assert b.val == 0
    assert b.len == 4
    assert b.as_bin() == '0000'


def test_bits_from_list_of_bools():
    b = Bits([True, False, True])
    assert b.val == 5
    assert b.len == 3
    assert b.as_list() == [True, False, True]


def test_bits_from_empty_list():
    b = Bits([])
    assert b.val == 0
    assert b.len == 0
    assert b.as_bin() == ''
    assert b.as_list() == []


def test_bits_from_bin_string():
    b = Bits('101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_empty_bin_string():
    b = Bits('')
    assert b.val == 0
    assert b.len == 0


def test_bits_from_bytes():
    b = Bits(b'101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_hex_string():
    b = Bits('0xFF')
    assert b.val == 255
    assert b.len == 8


def test_bits_from_hex_string_len_derived():
    b = Bits('0x0F')
    assert b.val == 15
    assert b.len == 8  # (len('0x0F') - 2) * 4


def test_bits_negative_raises():
    with pytest.raises(ValueError):
        Bits(-1)


def test_bits_bad_type():
    with pytest.raises(TypeError):
        Bits(1.5)
    with pytest.raises(TypeError):
        Bits(None)


# ---------------------------------------------------------------------------
# Bits: access
# ---------------------------------------------------------------------------

def test_bits_getitem():
    b = Bits('1010')
    assert b[0] is True
    assert b[1] is False
    assert b[2] is True
    assert b[3] is False


def test_bits_getitem_negative():
    b = Bits('1010')
    assert b[-1] is False
    assert b[-4] is True


def test_bits_getitem_out_of_range():
    b = Bits('1010')
    with pytest.raises(IndexError):
        b[4]
    with pytest.raises(IndexError):
        b[-5]


def test_bits_getitem_bad_type():
    b = Bits('1010')
    with pytest.raises(TypeError):
        b['a']


def test_bits_getitem_slice():
    b = Bits('1010')
    s = b[1:3]
    assert isinstance(s, Bits)
    assert s.as_bin() == '01'


def test_bits_len():
    assert len(Bits('10101')) == 5
    assert len(Bits([])) == 0


# ---------------------------------------------------------------------------
# Bits: equality / hashing
# ---------------------------------------------------------------------------

def test_bits_eq():
    assert Bits('101') == Bits('101')
    assert Bits(5, 3) == Bits('101')
    assert not (Bits('101') == Bits('1010'))
    assert Bits('1') != Bits('10')


def test_bits_eq_other_type():
    b = Bits('101')
    assert (b == 5) is False
    assert (b == '101') is False


def test_bits_hash():
    assert hash(Bits('101')) == hash(5)
    d = {Bits('101'): 'x'}
    assert d[Bits(5, 3)] == 'x'


# ---------------------------------------------------------------------------
# Bits: bitwise ops
# ---------------------------------------------------------------------------

def test_bits_or():
    a = Bits('1010')
    b = Bits('0110')
    r = a | b
    assert isinstance(r, Bits)
    assert r.val == (0b1010 | 0b0110)
    assert r.len == 4


def test_bits_or_different_lengths():
    a = Bits('1')       # len 1
    b = Bits('1010')    # len 4
    r = a | b
    assert r.len == 4
    assert r.val == (0b1 | 0b1010)


def test_bits_and():
    a = Bits('1010')
    b = Bits('0110')
    r = a & b
    assert r.val == (0b1010 & 0b0110)
    assert r.len == 4


def test_bits_and_different_lengths():
    a = Bits('1')
    b = Bits('1010')
    r = a & b
    assert r.len == 4
    assert r.val == (0b1 & 0b1010)


def test_bits_or_other_type():
    b = Bits('101')
    with pytest.raises(TypeError):
        b | 1


def test_bits_lshift():
    b = Bits('101')
    r = b << 2
    assert isinstance(r, Bits)
    assert r.val == (0b101 << 2)
    assert r.len == b.len + 2


def test_bits_rshift():
    b = Bits('1010')
    r = b >> 2
    assert r.val == (0b1010 >> 2)
    assert r.len == b.len - 2


# ---------------------------------------------------------------------------
# Bits: representations
# ---------------------------------------------------------------------------

def test_bits_as_list():
    assert Bits('1010').as_list() == [True, False, True, False]
    assert Bits(0, 4).as_list() == [False, False, False, False]


def test_bits_as_bin():
    assert Bits('101').as_bin() == '101'
    assert Bits(5, 8).as_bin() == '00000101'
    assert Bits([]).as_bin() == ''


def test_bits_as_hex():
    assert Bits('11111111').as_hex() == 'FF'
    assert Bits(0, 8).as_hex() == '00'
    assert Bits(255, 8).as_hex() == 'FF'


def test_bits_as_hex_padding():
    # 4 bits should pad to a single byte -> 2 hex chars
    assert Bits('1111').as_hex() == '0F'


def test_bits_as_hex_empty():
    assert Bits([]).as_hex() == ''


def test_bits_as_int():
    assert Bits('101').as_int() == 5
    assert Bits(0, 4).as_int() == 0


def test_bits_as_bytes():
    assert Bits('11111111').as_bytes() == b'\xff'
    assert Bits('0F', 8).as_bytes() == b'\x0f'


# ---------------------------------------------------------------------------
# Bits: classmethod constructors
# ---------------------------------------------------------------------------

def test_bits_from_list():
    b = Bits.from_list([True, False, True])
    assert b.val == 5
    assert b.len == 3


def test_bits_from_bin():
    b = Bits.from_bin('101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_hex():
    b = Bits.from_hex('FF')
    assert b.val == 255
    assert b.len == 8


def test_bits_from_hex_with_prefix():
    b = Bits.from_hex('0xFF')
    assert b.val == 255
    assert b.len == 8


def test_bits_from_hex_bytes():
    b = Bits.from_hex(b'FF')
    assert b.val == 255
    assert b.len == 8


def test_bits_from_hex_empty():
    b = Bits.from_hex('')
    assert b.val == 0
    assert b.len == 0


def test_bits_from_int_classmethod():
    b = Bits.from_int(5, 4)
    assert b.val == 5
    assert b.len == 4
    b2 = Bits.from_int(5)
    assert b2.val == 5
    assert b2.len == 3


def test_bits_from_bytes():
    b = Bits.from_bytes(b'\xff')
    assert b.val == 255
    assert b.len == 8


def test_bits_from_bytes_roundtrip():
    data = b'\x01\x02'
    b = Bits.from_bytes(data)
    assert b.as_bytes() == data


# ---------------------------------------------------------------------------
# Bits: repr
# ---------------------------------------------------------------------------

def test_bits_repr():
    b = Bits('101')
    assert repr(b) == "Bits('101')"


def test_bits_repr_empty():
    assert repr(Bits([])) == "Bits('')"
