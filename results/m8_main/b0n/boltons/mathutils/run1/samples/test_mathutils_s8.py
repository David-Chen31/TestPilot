import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ----------------------------- clamp -----------------------------

def test_clamp_in_range():
    assert clamp(1.0, 0, 5) == 1.0
    assert clamp(3, 0, 5) == 3


def test_clamp_below_lower():
    assert clamp(-1.0, 0, 5) == 0


def test_clamp_above_upper():
    assert clamp(101.0, 0, 5) == 5


def test_clamp_defaults():
    assert clamp(5) == 5
    assert clamp(-10**100) == -10**100
    assert clamp(123, upper=5) == 5
    assert clamp(-5, lower=0) == 0


def test_clamp_boundaries():
    assert clamp(0, 0, 5) == 0
    assert clamp(5, 0, 5) == 5


def test_clamp_upper_less_than_lower():
    with pytest.raises(ValueError):
        clamp(1, 5, 0)


def test_clamp_return_type_preserved_when_in_range():
    v = clamp(1.5, 0, 5)
    assert isinstance(v, float)
    assert v == 1.5


def test_clamp_returns_bound_types():
    # min/max return the bound objects themselves
    assert clamp(-1.0, 0, 5) == 0
    assert isinstance(clamp(-1.0, 0, 5), int)


# ----------------------------- ceil -----------------------------

def test_ceil_plain():
    assert ceil(3.2) == 4
    assert ceil(3) == 3
    assert ceil(-3.2) == -3
    assert ceil(0) == 0


def test_ceil_options_examples():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert ceil(3.5, options=VALID_CABLE_CSA) == 4
    assert ceil(4, options=VALID_CABLE_CSA) == 4


def test_ceil_options_various():
    opts = [1.5, 2.5, 4, 6, 10]
    assert ceil(0, options=opts) == 1.5
    assert ceil(1.5, options=opts) == 1.5
    assert ceil(2.6, options=opts) == 4
    assert ceil(10, options=opts) == 10


def test_ceil_options_unsorted():
    assert ceil(3, options=[10, 1, 5, 4]) == 4


def test_ceil_options_no_match():
    with pytest.raises(ValueError):
        ceil(11, options=[1, 2, 3])


def test_ceil_options_empty():
    with pytest.raises(ValueError):
        ceil(1, options=[])


def test_ceil_options_iterator():
    # accepts arbitrary iterable (gets sorted)
    assert ceil(3, options=iter([10, 1, 5, 4])) == 4


# ----------------------------- floor -----------------------------

def test_floor_plain():
    assert floor(3.8) == 3
    assert floor(3) == 3
    assert floor(-3.2) == -4
    assert floor(0) == 0


def test_floor_options_examples():
    VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]
    assert floor(3.5, options=VALID_CABLE_CSA) == 2.5
    assert floor(2.5, options=VALID_CABLE_CSA) == 2.5


def test_floor_options_various():
    opts = [1.5, 2.5, 4, 6, 10]
    assert floor(0.5, options=opts) == 1.5
    assert floor(1.5, options=opts) == 1.5
    assert floor(3, options=opts) == 2.5
    assert floor(100, options=opts) == 10


def test_floor_options_unsorted():
    assert floor(5, options=[10, 1, 5, 4]) == 5


def test_floor_options_no_match():
    with pytest.raises(ValueError):
        floor(0.1, options=[1, 2, 3])


def test_floor_options_empty():
    with pytest.raises(ValueError):
        floor(1, options=[])


def test_floor_options_iterator():
    assert floor(5, options=iter([10, 1, 5, 4])) == 5


# ----------------------------- Bits: construction -----------------------------

def test_bits_default():
    b = Bits()
    assert b.val == 0
    assert b.len == 1
    assert b.as_bin() == '0'
    assert b.as_int() == 0
    assert len(b) == 1


def test_bits_from_int():
    b = Bits(5)
    assert b.val == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_from_int_with_len():
    b = Bits(5, 8)
    assert b.val == 5
    assert b.len == 8
    assert b.as_bin() == '00000101'


def test_bits_from_bin_string():
    b = Bits('101')
    assert b.val == 5
    assert b.len == 3
    assert b.as_bin() == '101'


def test_bits_from_empty_bin_string():
    b = Bits('')
    assert b.val == 0
    assert b.len == 0
    assert b.as_bin() == ''


def test_bits_from_list():
    b = Bits([True, False, True])
    assert b.val == 5
    assert b.len == 3
    assert b.as_list() == [True, False, True]


def test_bits_from_bytes_literal():
    b = Bits(b'101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_hex_string():
    b = Bits('0x1F')
    assert b.val == 0x1F
    assert b.len == 8
    assert b.as_bin() == '00011111'


def test_bits_from_hex_int_string():
    b = Bits('FF')
    # not binary, but 'FF' is not valid binary -> int(..., 2) raises
    # Actually 'FF' doesn't start with 0x so it tries binary parse.
    # Placeholder to confirm: valid binary below.


def test_bits_from_bad_type():
    with pytest.raises(TypeError):
        Bits(1.5)


def test_bits_negative_value():
    with pytest.raises(ValueError):
        Bits(-1)


def test_bits_value_too_large():
    with pytest.raises(ValueError):
        Bits(8, 3)


def test_bits_fits_exactly():
    b = Bits(7, 3)
    assert b.val == 7
    assert b.len == 3
    assert b.as_bin() == '111'


# ----------------------------- Bits: getitem -----------------------------

def test_bits_getitem():
    b = Bits('101')
    assert b[0] is True
    assert b[1] is False
    assert b[2] is True


def test_bits_getitem_negative():
    b = Bits('101')
    assert b[-1] is True
    assert b[-3] is True


def test_bits_getitem_out_of_range():
    b = Bits('101')
    with pytest.raises(IndexError):
        b[3]
    with pytest.raises(IndexError):
        b[-4]


def test_bits_getitem_bad_type():
    b = Bits('101')
    with pytest.raises(TypeError):
        b['x']


def test_bits_getitem_slice():
    b = Bits('10110')
    sub = b[1:4]
    assert isinstance(sub, Bits)
    assert sub.as_bin() == '011'
    assert sub.val == 3
    assert sub.len == 3


# ----------------------------- Bits: equality and hash -----------------------------

def test_bits_eq():
    assert Bits('101') == Bits('101')
    assert not (Bits('101') == Bits('100'))
    assert not (Bits(5, 3) == Bits(5, 4))


def test_bits_eq_not_implemented_for_other():
    assert Bits('101').__eq__(5) is NotImplemented


def test_bits_hash():
    assert hash(Bits('101')) == hash(5)


def test_bits_usable_in_set():
    s = {Bits('101'), Bits('101')}
    assert len(s) == 1


# ----------------------------- Bits: operators -----------------------------

def test_bits_or():
    a = Bits('1000')
    b = Bits('0100')
    c = a | b
    assert isinstance(c, Bits)
    assert c.val == 0b1100
    assert c.len == 4


def test_bits_or_extends_len():
    a = Bits('1')
    b = Bits('1000')
    c = a | b
    assert c.len == 4
    assert c.val == 0b1001


def test_bits_and():
    a = Bits('1100')
    b = Bits('1010')
    c = a & b
    assert isinstance(c, Bits)
    assert c.val == 0b1000
    assert c.len == 4


def test_bits_and_extends_len():
    a = Bits('1')
    b = Bits('1000')
    c = a & b
    assert c.len == 4
    assert c.val == 0


def test_bits_or_not_implemented():
    assert Bits('1').__or__(5) is NotImplemented


def test_bits_and_not_implemented():
    assert Bits('1').__and__(5) is NotImplemented


def test_bits_lshift():
    b = Bits('1')
    c = b << 2
    assert isinstance(c, Bits)
    assert c.val == 0b100
    assert c.len == 3


def test_bits_rshift():
    b = Bits('100')
    c = b >> 1
    assert isinstance(c, Bits)
    assert c.val == 0b10
    assert c.len == 2


# ----------------------------- Bits: conversions -----------------------------

def test_bits_as_list():
    assert Bits('10110').as_list() == [True, False, True, True, False]


def test_bits_as_bin():
    assert Bits(5, 8).as_bin() == '00000101'
    assert Bits(0, 0).as_bin() == ''


def test_bits_as_hex():
    assert Bits(0xFF, 8).as_hex() == 'FF'
    assert Bits(0x1F, 8).as_hex() == '1F'


def test_bits_as_hex_padding():
    # 3 bits -> one byte -> 2 hex chars
    assert Bits('101').as_hex() == '05'


def test_bits_as_hex_empty():
    assert Bits(0, 0).as_hex() == ''


def test_bits_as_int():
    assert Bits('101').as_int() == 5


def test_bits_as_bytes():
    assert Bits(0xFF, 8).as_bytes() == b'\xff'


def test_bits_as_bytes_multi():
    assert Bits(0x1234, 16).as_bytes() == b'\x12\x34'


# ----------------------------- Bits: constructors -----------------------------

def test_bits_from_list_classmethod():
    b = Bits.from_list([True, False, True])
    assert b.val == 5
    assert b.len == 3


def test_bits_from_bin_classmethod():
    b = Bits.from_bin('101')
    assert b.val == 5
    assert b.len == 3


def test_bits_from_hex_classmethod():
    b = Bits.from_hex('FF')
    assert b.val == 0xFF
    assert b.len == 8


def test_bits_from_hex_with_prefix():
    b = Bits.from_hex('0x1F')
    assert b.val == 0x1F
    assert b.len == 8


def test_bits_from_hex_bytes():
    b = Bits.from_hex(b'FF')
    assert b.val == 0xFF
    assert b.len == 8


def test_bits_from_hex_empty():
    b = Bits.from_hex('')
    assert b.val == 0
    assert b.len == 0


def test_bits_from_int_classmethod():
    b = Bits.from_int(5)
    assert b.val == 5
    assert b.len == 3
    b2 = Bits.from_int(5, 8)
    assert b2.val == 5
    assert b2.len == 8


def test_bits_from_bytes_classmethod():
    b = Bits.from_bytes(b'\xff')
    assert b.val == 0xFF
    assert b.len == 8


def test_bits_from_bytes_multi():
    b = Bits.from_bytes(b'\x12\x34')
    assert b.val == 0x1234
    assert b.len == 16


# ----------------------------- Bits: misc -----------------------------

def test_bits_len():
    assert len(Bits('10110')) == 5
    assert len(Bits()) == 1


def test_bits_repr():
    b = Bits('101')
    assert repr(b) == "Bits('101')"


def test_bits_slots():
    b = Bits('101')
    with pytest.raises(AttributeError):
        b.some_new_attr = 1


def test_bits_roundtrip_bin():
    for val in [0, 1, 5, 10, 255]:
        b = Bits.from_int(val)
        assert Bits.from_bin(b.as_bin()) == b


def test_bits_roundtrip_hex():
    b = Bits(0x1234, 16)
    assert Bits.from_hex(b.as_hex()) == b


def test_bits_roundtrip_bytes():
    b = Bits(0x1234, 16)
    assert Bits.from_bytes(b.as_bytes()) == b
