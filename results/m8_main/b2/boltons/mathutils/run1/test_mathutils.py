import pytest
import pickle

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
        assert clamp(-100, 0, 5) == 0

    def test_above_upper(self):
        assert clamp(101.0, 0, 5) == 5
        assert clamp(123, upper=5) == 5

    def test_exact_bounds(self):
        assert clamp(0, 0, 5) == 0
        assert clamp(5, 0, 5) == 5

    def test_default_bounds(self):
        assert clamp(123) == 123
        assert clamp(-123) == -123
        assert clamp(0.5) == 0.5

    def test_lower_only(self):
        assert clamp(-5, lower=0) == 0
        assert clamp(5, lower=0) == 5

    def test_upper_only(self):
        assert clamp(10, upper=3) == 3
        assert clamp(-10, upper=3) == -10

    def test_equal_bounds(self):
        assert clamp(5, 3, 3) == 3
        assert clamp(1, 3, 3) == 3
        assert clamp(100, 3, 3) == 3

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            clamp(1, lower=5, upper=0)

    def test_invalid_bounds_message(self):
        with pytest.raises(ValueError) as excinfo:
            clamp(1, lower=5, upper=0)
        msg = str(excinfo.value)
        assert msg == 'expected upper bound (0) >= lower bound (5)'

    def test_mixed_types(self):
        assert clamp(1.5, 0, 2) == 1.5
        assert clamp(2.9, 0, 2) == 2

    def test_preserves_type_of_bound(self):
        # when clamped, the returned value is the bound itself
        result = clamp(-1.0, 0, 5)
        assert result == 0
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]


class TestCeil:
    def test_builtin_no_options(self):
        assert ceil(1.2) == 2
        assert ceil(1.0) == 1
        assert ceil(-1.2) == -1
        assert ceil(-1.0) == -1
        assert ceil(0) == 0
        assert ceil(-0.5) == 0

    def test_options_exact(self):
        assert ceil(3.5, options=VALID_CABLE_CSA) == 4
        assert ceil(4, options=VALID_CABLE_CSA) == 4
        assert ceil(2.5, options=VALID_CABLE_CSA) == 2.5

    def test_options_between(self):
        assert ceil(3.0, options=VALID_CABLE_CSA) == 4
        assert ceil(0.5, options=VALID_CABLE_CSA) == 1.5
        assert ceil(7, options=VALID_CABLE_CSA) == 10

    def test_options_above_max(self):
        with pytest.raises(ValueError):
            ceil(51, options=VALID_CABLE_CSA)
        with pytest.raises(ValueError):
            ceil(1000, options=VALID_CABLE_CSA)

    def test_options_above_max_message(self):
        with pytest.raises(ValueError) as excinfo:
            ceil(51, options=VALID_CABLE_CSA)
        assert str(excinfo.value) == \
            "no ceil options greater than or equal to: 51"

    def test_options_unsorted(self):
        opts = [10, 1.5, 6, 2.5, 50, 4, 35, 25]
        assert ceil(3.0, options=opts) == 4
        assert ceil(4, options=opts) == 4

    def test_options_iterable(self):
        # generator / non-list iterable should work (sorted internally)
        result = ceil(3.0, options=(x for x in VALID_CABLE_CSA))
        assert result == 4

    def test_options_empty(self):
        with pytest.raises(ValueError):
            ceil(3.0, options=[])

    def test_options_empty_message(self):
        with pytest.raises(ValueError) as excinfo:
            ceil(3.0, options=[])
        assert str(excinfo.value) == \
            "no ceil options greater than or equal to: 3.0"


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

class TestFloor:
    def test_builtin_no_options(self):
        assert floor(1.8) == 1
        assert floor(1.0) == 1
        assert floor(-1.2) == -2
        assert floor(-1.0) == -1
        assert floor(0) == 0
        assert floor(0.5) == 0

    def test_options_exact(self):
        assert floor(3.5, options=VALID_CABLE_CSA) == 2.5
        assert floor(2.5, options=VALID_CABLE_CSA) == 2.5
        assert floor(4, options=VALID_CABLE_CSA) == 4

    def test_options_between(self):
        assert floor(3.0, options=VALID_CABLE_CSA) == 2.5
        assert floor(9.999, options=VALID_CABLE_CSA) == 6
        assert floor(100, options=VALID_CABLE_CSA) == 50

    def test_options_below_min(self):
        with pytest.raises(ValueError):
            floor(0.1, options=VALID_CABLE_CSA)
        with pytest.raises(ValueError):
            floor(-100, options=VALID_CABLE_CSA)

    def test_options_below_min_message(self):
        with pytest.raises(ValueError) as excinfo:
            floor(0.1, options=VALID_CABLE_CSA)
        assert str(excinfo.value) == \
            "no floor options less than or equal to: 0.1"

    def test_options_unsorted(self):
        opts = [10, 1.5, 6, 2.5, 50, 4, 35, 25]
        assert floor(3.0, options=opts) == 2.5
        assert floor(4, options=opts) == 4

    def test_options_iterable(self):
        result = floor(3.0, options=(x for x in VALID_CABLE_CSA))
        assert result == 2.5

    def test_options_empty(self):
        with pytest.raises(ValueError):
            floor(3.0, options=[])

    def test_options_empty_message(self):
        with pytest.raises(ValueError) as excinfo:
            floor(3.0, options=[])
        assert str(excinfo.value) == \
            "no floor options less than or equal to: 3.0"


# ---------------------------------------------------------------------------
# Bits: construction
# ---------------------------------------------------------------------------

class TestBitsConstruction:
    def test_default(self):
        b = Bits()
        assert b.val == 0
        assert b.len == 1
        assert b.as_bin() == '0'
        assert len(b) == 1

    def test_from_int(self):
        b = Bits(5)
        assert b.val == 5
        assert b.len == 3
        assert b.as_bin() == '101'

    def test_from_int_with_len(self):
        b = Bits(5, 8)
        assert b.val == 5
        assert b.len == 8
        assert b.as_bin() == '00000101'

    def test_from_bin_str(self):
        b = Bits('1011')
        assert b.val == 11
        assert b.len == 4
        assert b.as_bin() == '1011'

    def test_from_bin_leading_zeros(self):
        b = Bits('0011')
        assert b.val == 3
        assert b.len == 4
        assert b.as_bin() == '0011'

    def test_from_empty_str(self):
        b = Bits('')
        assert b.val == 0
        assert b.len == 0
        assert b.as_bin() == ''

    def test_from_hex_str(self):
        b = Bits('0xff')
        assert b.val == 255
        assert b.len == 8
        assert b.as_hex() == 'FF'
        assert b.as_bin() == '11111111'

    def test_from_hex_with_len(self):
        b = Bits('0x0f', 8)
        assert b.val == 15
        assert b.len == 8
        assert b.as_bin() == '00001111'

    def test_from_bytes(self):
        b = Bits(b'1010')
        assert b.val == 10
        assert b.len == 4

    def test_from_hex_uppercase_bytes(self):
        # exercises the bytes -> str decode path
        b = Bits(b'0xFF')
        assert b.val == 255
        assert b.len == 8

    def test_from_list(self):
        b = Bits([True, False, True, True])
        assert b.val == 11
        assert b.len == 4
        assert b.as_list() == [True, False, True, True]

    def test_from_empty_list(self):
        b = Bits([])
        assert b.val == 0
        assert b.len == 0

    def test_negative_value(self):
        with pytest.raises(ValueError):
            Bits(-1)

    def test_negative_value_message(self):
        with pytest.raises(ValueError) as excinfo:
            Bits(-1)
        assert str(excinfo.value) == 'Bits cannot represent negative values'

    def test_value_too_large_for_len(self):
        with pytest.raises(ValueError):
            Bits(8, 3)
        with pytest.raises(ValueError):
            Bits(9, 3)

    def test_value_too_large_message(self):
        with pytest.raises(ValueError) as excinfo:
            Bits(8, 3)
        assert str(excinfo.value) == 'value 8 cannot be represented with 3 bits'

    def test_value_fits(self):
        b = Bits(7, 3)
        assert b.val == 7
        assert b.len == 3

    def test_bad_type(self):
        with pytest.raises(TypeError):
            Bits(1.5)
        with pytest.raises(TypeError):
            Bits(object())

    def test_bad_type_message(self):
        with pytest.raises(TypeError) as excinfo:
            Bits(1.5)
        assert str(excinfo.value) == 'initialized with bad type: float'

    def test_zero_len_int(self):
        b = Bits(0, 0)
        assert b.val == 0
        assert b.len == 0


# ---------------------------------------------------------------------------
# Bits: access / length / getitem
# ---------------------------------------------------------------------------

class TestBitsAccess:
    def test_len(self):
        assert len(Bits('101')) == 3
        assert len(Bits(0, 0)) == 0

    def test_getitem_positive(self):
        b = Bits('1011')
        assert b[0] is True
        assert b[1] is False
        assert b[2] is True
        assert b[3] is True

    def test_getitem_negative(self):
        b = Bits('1011')
        assert b[-1] is True
        assert b[-2] is True
        assert b[-3] is False
        assert b[-4] is True

    def test_getitem_index_error(self):
        b = Bits('1011')
        with pytest.raises(IndexError):
            b[4]
        with pytest.raises(IndexError):
            b[-5]

    def test_getitem_bad_type(self):
        b = Bits('1011')
        with pytest.raises(TypeError):
            b['a']
        with pytest.raises(TypeError):
            b[1.5]

    def test_getitem_slice(self):
        b = Bits('10110011')
        sliced = b[2:6]
        assert isinstance(sliced, Bits)
        assert sliced.as_bin() == '1100'

    def test_getitem_slice_full(self):
        b = Bits('101')
        sliced = b[:]
        assert sliced.as_bin() == '101'

    def test_getitem_slice_empty(self):
        b = Bits('1011')
        sliced = b[2:2]
        assert isinstance(sliced, Bits)
        assert sliced.as_bin() == ''
        assert sliced.len == 0


# ---------------------------------------------------------------------------
# Bits: equality / hash
# ---------------------------------------------------------------------------

class TestBitsEqHash:
    def test_eq_same(self):
        assert Bits('101') == Bits('101')
        assert Bits(5, 3) == Bits('101')

    def test_eq_different_val(self):
        assert Bits('101') != Bits('110')

    def test_eq_different_len(self):
        # same value but different lengths are not equal
        assert Bits(5, 3) != Bits(5, 8)
        assert Bits('101') != Bits('0101')

    def test_eq_other_type(self):
        assert (Bits('101') == 5) is False
        assert (Bits('101') == '101') is False

    def test_hash(self):
        assert hash(Bits('101')) == hash(Bits(5))
        d = {Bits('101'): 'a'}
        assert d[Bits(5)] == 'a'
        # different lengths, same value -> hashes can collide, that's fine
        assert hash(Bits('101')) == hash(Bits('0101'))


# ---------------------------------------------------------------------------
# Bits: bitwise operators
# ---------------------------------------------------------------------------

class TestBitsOperators:
    def test_or(self):
        result = Bits('1010') | Bits('0110')
        assert isinstance(result, Bits)
        assert result.val == 0b1110
        assert result.len == 4

    def test_or_different_lengths(self):
        result = Bits('101') | Bits('00001')
        assert result.val == 0b00101
        assert result.len == 5

    def test_and(self):
        result = Bits('1010') & Bits('0110')
        assert result.val == 0b0010
        assert result.len == 4

    def test_and_different_lengths(self):
        result = Bits('101') & Bits('00101')
        assert result.val == 0b00101 & 0b00101
        assert result.len == 5

    def test_lshift(self):
        result = Bits('101') << 2
        assert isinstance(result, Bits)
        assert result.val == 0b10100
        assert result.len == 5
        assert result.as_bin() == '10100'

    def test_rshift(self):
        result = Bits('10100') >> 2
        assert result.val == 0b101
        assert result.len == 3
        assert result.as_bin() == '101'

    def test_or_not_implemented(self):
        with pytest.raises(TypeError):
            Bits('101') | 5

    def test_and_not_implemented(self):
        with pytest.raises(TypeError):
            Bits('101') & 5


# ---------------------------------------------------------------------------
# Bits: conversions
# ---------------------------------------------------------------------------

class TestBitsConversions:
    def test_as_list(self):
        assert Bits('1011').as_list() == [True, False, True, True]

    def test_as_list_empty(self):
        assert Bits('').as_list() == []

    def test_as_bin(self):
        assert Bits('101').as_bin() == '101'
        assert Bits(255).as_bin() == '11111111'
        assert Bits(5, 8).as_bin() == '00000101'

    def test_as_bin_empty(self):
        assert Bits('').as_bin() == ''
        assert Bits(0, 0).as_bin() == ''

    def test_as_hex(self):
        assert Bits('11111111').as_hex() == 'FF'
        # length 4 -> 1 byte, padded to 2 hex chars
        assert Bits(15).as_hex() == '0F'
        assert Bits('1111').as_hex() == '0F'
        assert Bits(5, 8).as_hex() == '05'

    def test_as_hex_empty(self):
        assert Bits('').as_hex() == ''

    def test_as_hex_multi_byte(self):
        assert Bits('1' * 16).as_hex() == 'FFFF'
        assert Bits(0x1234, 16).as_hex() == '1234'

    def test_as_int(self):
        assert Bits('101').as_int() == 5
        assert Bits('').as_int() == 0

    def test_as_bytes(self):
        assert Bits('11111111').as_bytes() == b'\xff'
        assert Bits(0x1234, 16).as_bytes() == b'\x12\x34'

    def test_as_bytes_empty(self):
        assert Bits('').as_bytes() == b''

    def test_roundtrip_bytes(self):
        original = b'\xde\xad'
        b = Bits.from_bytes(original)
        assert b.as_bytes() == original


# ---------------------------------------------------------------------------
# Bits: factory classmethods
# ---------------------------------------------------------------------------

class TestBitsClassmethods:
    def test_from_list(self):
        b = Bits.from_list([True, False, True])
        assert b.val == 0b101
        assert b.len == 3

    def test_from_bin(self):
        b = Bits.from_bin('1010')
        assert b.val == 10
        assert b.len == 4

    def test_from_hex_plain(self):
        b = Bits.from_hex('ff')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_prefixed(self):
        b = Bits.from_hex('0xff')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_bytes(self):
        b = Bits.from_hex(b'ff')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_empty(self):
        b = Bits.from_hex('')
        assert b.val == 0
        assert b.len == 0

    def test_from_hex_empty_bytes(self):
        b = Bits.from_hex(b'')
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
        assert b.as_bin() == '00000101'

    def test_from_bytes(self):
        b = Bits.from_bytes(b'\x00\xff')
        assert b.val == 255
        assert b.len == 16
        assert b.as_hex() == '00FF'


# ---------------------------------------------------------------------------
# Bits: repr
# ---------------------------------------------------------------------------

class TestBitsRepr:
    def test_repr(self):
        b = Bits('1011')
        assert repr(b) == "Bits('1011')"

    def test_repr_empty(self):
        assert repr(Bits('')) == "Bits('')"
