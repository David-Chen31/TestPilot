import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert clamp(3, 0, 5) == 3
        assert clamp(1.0, 0, 5) == 1.0

    def test_below_lower(self):
        assert clamp(-1.0, 0, 5) == 0
        assert clamp(-100, 0, 5) == 0

    def test_above_upper(self):
        assert clamp(101.0, 0, 5) == 5
        assert clamp(123, upper=5) == 5

    def test_default_bounds(self):
        assert clamp(12345) == 12345
        assert clamp(-12345) == -12345

    def test_equal_bounds(self):
        assert clamp(5, 5, 5) == 5
        assert clamp(0, 5, 5) == 5
        assert clamp(10, 5, 5) == 5

    def test_upper_less_than_lower_raises(self):
        with pytest.raises(ValueError):
            clamp(1, 5, 0)

    def test_mixed_types(self):
        assert clamp(1.5, 1, 2) == 1.5
        assert clamp(3, 0.0, 5.0) == 3


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]


class TestCeil:
    def test_no_options(self):
        assert ceil(3.5) == 4
        assert ceil(4) == 4
        assert ceil(4.0001) == 5
        assert ceil(-1.5) == -1
        assert ceil(-2.0) == -2

    def test_options_basic(self):
        assert ceil(3.5, options=VALID_CABLE_CSA) == 4
        assert ceil(4, options=VALID_CABLE_CSA) == 4
        assert ceil(2.5, options=VALID_CABLE_CSA) == 2.5

    def test_options_exact_and_below(self):
        assert ceil(1.5, options=VALID_CABLE_CSA) == 1.5
        assert ceil(0, options=VALID_CABLE_CSA) == 1.5

    def test_options_above_max(self):
        assert ceil(50, options=VALID_CABLE_CSA) == 50
        with pytest.raises(ValueError):
            ceil(51, options=VALID_CABLE_CSA)

    def test_options_unsorted(self):
        assert ceil(3.5, options=[10, 4, 1.5]) == 4

    def test_options_empty(self):
        with pytest.raises(ValueError):
            ceil(1, options=[])

    def test_options_tuple(self):
        assert ceil(3, options=(1, 2, 4, 8)) == 4


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

class TestFloor:
    def test_no_options(self):
        assert floor(3.5) == 3
        assert floor(4) == 4
        assert floor(3.999) == 3
        assert floor(-1.5) == -2

    def test_options_basic(self):
        assert floor(3.5, options=VALID_CABLE_CSA) == 2.5
        assert floor(2.5, options=VALID_CABLE_CSA) == 2.5
        assert floor(4, options=VALID_CABLE_CSA) == 4

    def test_options_exact(self):
        assert floor(1.5, options=VALID_CABLE_CSA) == 1.5
        assert floor(50, options=VALID_CABLE_CSA) == 50

    def test_options_below_min(self):
        with pytest.raises(ValueError):
            floor(0.5, options=VALID_CABLE_CSA)

    def test_options_unsorted(self):
        assert floor(3.5, options=[10, 4, 1.5]) == 1.5

    def test_options_empty(self):
        with pytest.raises(ValueError):
            floor(1, options=[])

    def test_options_tuple(self):
        assert floor(3, options=(1, 2, 4, 8)) == 2


# ---------------------------------------------------------------------------
# Bits
# ---------------------------------------------------------------------------

class TestBitsConstruction:
    def test_default(self):
        b = Bits()
        assert b.val == 0
        assert b.len == 1
        assert b.as_bin() == '0'

    def test_from_int(self):
        b = Bits(5)
        assert b.val == 5
        assert b.len == 3
        assert b.as_int() == 5

    def test_from_int_with_len(self):
        b = Bits(5, 8)
        assert b.val == 5
        assert b.len == 8
        assert b.as_bin() == '00000101'

    def test_from_list(self):
        b = Bits([True, False, True])
        assert b.val == 5
        assert b.len == 3
        assert b.as_bin() == '101'

    def test_from_bin_string(self):
        b = Bits('101')
        assert b.val == 5
        assert b.len == 3

    def test_from_hex_string(self):
        b = Bits('0xFF')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_string_short(self):
        b = Bits('0x1')
        assert b.val == 1
        assert b.len == 4

    def test_from_empty_string(self):
        b = Bits('')
        assert b.val == 0
        assert b.len == 0

    def test_from_bytes(self):
        b = Bits(b'101')
        assert b.val == 5
        assert b.len == 3

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            Bits(-1)

    def test_value_too_large_raises(self):
        with pytest.raises(ValueError):
            Bits(5, 2)

    def test_value_exact_max(self):
        b = Bits(3, 2)
        assert b.val == 3
        assert b.len == 2

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            Bits(1.5)
        with pytest.raises(TypeError):
            Bits(object())

    def test_zero_len(self):
        b = Bits(0, 0)
        assert b.val == 0
        assert b.len == 0
        assert b.as_bin() == ''
        assert b.as_hex() == ''
        assert len(b) == 0


class TestBitsAccess:
    def test_len(self):
        assert len(Bits(5, 3)) == 3

    def test_getitem_int(self):
        b = Bits('101')
        assert b[0] is True
        assert b[1] is False
        assert b[2] is True

    def test_getitem_negative(self):
        b = Bits('101')
        assert b[-1] is True
        assert b[-2] is False
        assert b[-3] is True

    def test_getitem_int_out_of_range(self):
        b = Bits('101')
        with pytest.raises(IndexError):
            b[3]
        with pytest.raises(IndexError):
            b[-4]

    def test_getitem_slice(self):
        b = Bits('10110')
        sliced = b[1:4]
        assert isinstance(sliced, Bits)
        assert sliced.as_bin() == '011'

    def test_getitem_bad_type(self):
        b = Bits('101')
        with pytest.raises(TypeError):
            b['a']


class TestBitsRepresentations:
    def test_as_bin(self):
        assert Bits(5, 3).as_bin() == '101'
        assert Bits(5, 8).as_bin() == '00000101'

    def test_as_list(self):
        assert Bits('101').as_list() == [True, False, True]

    def test_as_int(self):
        assert Bits('101').as_int() == 5

    def test_as_hex(self):
        assert Bits('00000101').as_hex() == '05'
        assert Bits(255, 8).as_hex() == 'FF'

    def test_as_hex_padding(self):
        b = Bits(1, 1)
        assert b.as_hex() == '1'
        b = Bits(1, 9)
        assert b.as_hex() == '0001'

    def test_as_bytes(self):
        assert Bits(255, 8).as_bytes() == b'\xff'
        assert Bits('00000101').as_bytes() == b'\x05'

    def test_as_bytes_empty(self):
        assert Bits(0, 0).as_bytes() == b''


class TestBitsClassMethods:
    def test_from_list(self):
        b = Bits.from_list([True, False, True])
        assert b.as_bin() == '101'

    def test_from_bin(self):
        b = Bits.from_bin('101')
        assert b.val == 5
        assert b.len == 3

    def test_from_hex(self):
        b = Bits.from_hex('FF')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_with_prefix(self):
        b = Bits.from_hex('0xFF')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_bytes(self):
        b = Bits.from_hex(b'FF')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_empty(self):
        b = Bits.from_hex('')
        assert b.val == 0
        assert b.len == 0

    def test_from_int(self):
        b = Bits.from_int(5)
        assert b.val == 5
        assert b.len == 3

    def test_from_int_with_len(self):
        b = Bits.from_int(5, 8)
        assert b.val == 5
        assert b.len == 8

    def test_from_bytes(self):
        b = Bits.from_bytes(b'\x05')
        assert b.val == 5
        assert b.len == 8


class TestBitsEqualityAndHash:
    def test_eq_same(self):
        assert Bits('101') == Bits('101')

    def test_eq_different_val(self):
        assert Bits('101') != Bits('110')

    def test_eq_different_len(self):
        assert Bits(5, 3) != Bits(5, 4)

    def test_eq_other_type(self):
        assert Bits('101') != 5
        assert Bits('101') != '101'

    def test_hash(self):
        assert hash(Bits('101')) == hash(5)
        assert hash(Bits('101')) == hash(Bits('101'))

    def test_usable_in_set(self):
        s = {Bits('101'), Bits('101'), Bits('110')}
        assert len(s) == 2


class TestBitsOperators:
    def test_or(self):
        assert Bits('100') | Bits('010') == Bits('110')

    def test_or_diff_len(self):
        result = Bits('100') | Bits('10')
        assert result.len == 3
        assert result.val == 6

    def test_and(self):
        assert Bits('110') & Bits('011') == Bits('010')

    def test_and_diff_len(self):
        result = Bits('110') & Bits('11')
        assert result.len == 3
        assert result.val == 2

    def test_or_other_type(self):
        with pytest.raises(TypeError):
            Bits('101') | 5

    def test_and_other_type(self):
        with pytest.raises(TypeError):
            Bits('101') & 5

    def test_lshift(self):
        result = Bits('101') << 2
        assert result.val == 20
        assert result.len == 5
        assert result.as_bin() == '10100'

    def test_rshift(self):
        result = Bits('10100') >> 2
        assert result.val == 5
        assert result.len == 3
        assert result.as_bin() == '101'


class TestBitsRepr:
    def test_repr(self):
        assert repr(Bits('101')) == "Bits('101')"

    def test_repr_empty(self):
        assert repr(Bits(0, 0)) == "Bits('')"


class TestBitsImmutability:
    def test_slots(self):
        b = Bits('101')
        with pytest.raises(AttributeError):
            b.some_new_attr = 1
