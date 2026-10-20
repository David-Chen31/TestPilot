import pytest

from boltons.mathutils import clamp, ceil, floor, Bits


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert clamp(1.0, 0, 5) == 1.0

    def test_below_lower(self):
        assert clamp(-1.0, 0, 5) == 0

    def test_above_upper(self):
        assert clamp(101.0, 0, 5) == 5

    def test_only_upper(self):
        assert clamp(123, upper=5) == 5

    def test_only_lower(self):
        assert clamp(-123, lower=-5) == -5

    def test_defaults(self):
        assert clamp(42) == 42

    def test_defaults_large(self):
        assert clamp(10 ** 100) == 10 ** 100

    def test_lower_equal_upper(self):
        assert clamp(3, 5, 5) == 5

    def test_value_equal_bounds(self):
        assert clamp(5, 5, 5) == 5

    def test_returns_exact_type_int(self):
        # min/max returns the actual object, no coercion
        result = clamp(2, 0, 5)
        assert result == 2
        assert isinstance(result, int)

    def test_returns_bound_object(self):
        # when clamped, the returned object is the bound object
        upper = 5
        assert clamp(100, 0, upper) is upper

    def test_upper_less_than_lower_raises(self):
        with pytest.raises(ValueError):
            clamp(1, 10, 0)

    def test_upper_less_than_lower_error_message(self):
        with pytest.raises(ValueError) as excinfo:
            clamp(1, 10, 0)
        assert 'expected upper bound' in str(excinfo.value)

    def test_infinity_bounds(self):
        assert clamp(float('inf')) == float('inf')
        assert clamp(float('-inf')) == float('-inf')

    def test_lower_infinity(self):
        assert clamp(-10 ** 100, lower=float('-inf'), upper=0) == -10 ** 100

    def test_strings_comparable(self):
        assert clamp('b', 'a', 'c') == 'b'
        assert clamp('z', 'a', 'c') == 'c'


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]


class TestCeil:
    def test_plain_int(self):
        assert ceil(3) == 3

    def test_plain_float_round_up(self):
        assert ceil(3.2) == 4

    def test_plain_float_exact(self):
        assert ceil(3.0) == 3

    def test_plain_negative(self):
        assert ceil(-3.2) == -3

    def test_plain_negative_exact(self):
        assert ceil(-3.0) == -3

    def test_returns_int_for_plain(self):
        assert isinstance(ceil(3.2), int)

    def test_options_none_explicit(self):
        assert ceil(2.1, options=None) == 3

    def test_options_exact(self):
        assert ceil(3.5, options=VALID_CABLE_CSA) == 4

    def test_options_on_option(self):
        assert ceil(4, options=VALID_CABLE_CSA) == 4

    def test_options_between(self):
        assert ceil(1.6, options=VALID_CABLE_CSA) == 2.5

    def test_options_below_first(self):
        assert ceil(0.5, options=VALID_CABLE_CSA) == 1.5

    def test_options_above_last_raises(self):
        with pytest.raises(ValueError):
            ceil(100, options=VALID_CABLE_CSA)

    def test_options_unsorted_input(self):
        assert ceil(3.5, options=[50, 1.5, 25, 2.5, 10, 4, 6, 35]) == 4

    def test_options_single(self):
        assert ceil(5, options=[10]) == 10

    def test_options_does_not_mutate(self):
        opts = [50, 1.5, 25]
        try:
            ceil(2, options=opts)
        except ValueError:
            pass
        assert opts == [50, 1.5, 25]

    def test_options_empty_raises(self):
        with pytest.raises(ValueError):
            ceil(1, options=[])


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

class TestFloor:
    def test_plain_int(self):
        assert floor(3) == 3

    def test_plain_float_round_down(self):
        assert floor(3.8) == 3

    def test_plain_float_exact(self):
        assert floor(3.0) == 3

    def test_plain_negative(self):
        assert floor(-3.2) == -4

    def test_plain_negative_exact(self):
        assert floor(-3.0) == -3

    def test_returns_int_for_plain(self):
        assert isinstance(floor(3.8), int)

    def test_options_none_explicit(self):
        assert floor(2.9, options=None) == 2

    def test_options_between(self):
        assert floor(3.5, options=VALID_CABLE_CSA) == 2.5

    def test_options_on_option(self):
        assert floor(2.5, options=VALID_CABLE_CSA) == 2.5

    def test_options_on_integer_option(self):
        assert floor(6, options=VALID_CABLE_CSA) == 6

    def test_options_above_last(self):
        assert floor(100, options=VALID_CABLE_CSA) == 50

    def test_options_below_first_raises(self):
        with pytest.raises(ValueError):
            floor(0.5, options=VALID_CABLE_CSA)

    def test_options_unsorted_input(self):
        assert floor(3.5, options=[50, 1.5, 25, 2.5, 10, 4, 6, 35]) == 2.5

    def test_options_single(self):
        assert floor(5, options=[10]) == 10 if False else floor(15, options=[10]) == 10

    def test_options_empty_raises(self):
        with pytest.raises(ValueError):
            floor(1, options=[])


# ---------------------------------------------------------------------------
# Bits - construction
# ---------------------------------------------------------------------------

class TestBitsInit:
    def test_default(self):
        b = Bits()
        assert b.val == 0
        assert b.len == 1
        assert b.as_bin() == '0'

    def test_int_default_len(self):
        b = Bits(5)
        assert b.val == 5
        assert b.len == 3
        assert b.as_bin() == '101'

    def test_int_with_len(self):
        b = Bits(5, 8)
        assert b.val == 5
        assert b.len == 8
        assert b.as_bin() == '00000101'

    def test_int_value_too_big_raises(self):
        with pytest.raises(ValueError):
            Bits(16, 4)

    def test_int_value_fits_exactly(self):
        b = Bits(15, 4)
        assert b.val == 15
        assert b.len == 4

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            Bits(-1)

    def test_str_bin(self):
        b = Bits('101')
        assert b.val == 5
        assert b.len == 3

    def test_str_bin_leading_zeros(self):
        b = Bits('00101')
        assert b.val == 5
        assert b.len == 5
        assert b.as_bin() == '00101'

    def test_str_empty(self):
        b = Bits('')
        assert b.val == 0
        assert b.len == 0
        assert b.as_bin() == ''

    def test_str_hex(self):
        b = Bits('0xFF')
        assert b.val == 255
        assert b.len == 8

    def test_str_hex_lowercase(self):
        b = Bits('0xab')
        assert b.val == 0xab
        assert b.len == 8

    def test_str_hex_with_len(self):
        b = Bits('0xA', 8)
        assert b.val == 10
        assert b.len == 8
        assert b.as_bin() == '00001010'

    def test_bytes(self):
        b = Bits(b'101')
        assert b.val == 5
        assert b.len == 3

    def test_bytes_hex(self):
        b = Bits(b'0xF')
        assert b.val == 15
        assert b.len == 4

    def test_list_of_bools(self):
        b = Bits([True, False, True])
        assert b.val == 5
        assert b.len == 3

    def test_list_empty(self):
        b = Bits([])
        assert b.val == 0
        assert b.len == 0

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            Bits(1.5)

    def test_bad_type_object_raises(self):
        with pytest.raises(TypeError):
            Bits(object())


# ---------------------------------------------------------------------------
# Bits - getitem / setattr / len
# ---------------------------------------------------------------------------

class TestBitsGetitem:
    def test_getitem_index(self):
        b = Bits('101')
        assert b[0] is True
        assert b[1] is False
        assert b[2] is True

    def test_getitem_returns_bool(self):
        b = Bits('1')
        assert isinstance(b[0], bool)

    def test_getitem_negative(self):
        b = Bits('101')
        assert b[-1] is True
        assert b[-2] is False
        assert b[-3] is True

    def test_getitem_out_of_range(self):
        b = Bits('101')
        with pytest.raises(IndexError):
            b[3]

    def test_getitem_negative_out_of_range(self):
        b = Bits('101')
        with pytest.raises(IndexError):
            b[-4]

    def test_getitem_bad_type(self):
        b = Bits('101')
        with pytest.raises(TypeError):
            b['a']

    def test_getitem_slice(self):
        b = Bits('10110')
        s = b[1:4]
        assert isinstance(s, Bits)
        assert s.as_bin() == '011'
        assert s.val == 3
        assert s.len == 3

    def test_getitem_slice_full(self):
        b = Bits('1010')
        s = b[:]
        assert s.as_bin() == '1010'

    def test_getitem_slice_step(self):
        b = Bits('1010')
        s = b[::2]
        assert s.as_bin() == '11'

    def test_len(self):
        assert len(Bits('101')) == 3
        assert len(Bits('')) == 0
        assert len(Bits(5)) == 3


# ---------------------------------------------------------------------------
# Bits - equality / hash
# ---------------------------------------------------------------------------

class TestBitsEq:
    def test_eq_same(self):
        assert Bits('101') == Bits('101')

    def test_eq_different_val(self):
        assert Bits('101') != Bits('100')

    def test_eq_different_len(self):
        assert Bits('101') != Bits('0101')

    def test_eq_non_bits(self):
        assert Bits('101') != '101'
        assert Bits('101') != 5

    def test_eq_returns_notimplemented_leading_to_false(self):
        # __eq__ returns NotImplemented for non-Bits, so falls back to False
        result = Bits('1').__eq__(1)
        assert result is NotImplemented

    def test_ne(self):
        assert not (Bits('101') == Bits('10'))

    def test_hash(self):
        assert hash(Bits('101')) == hash(5)

    def test_hash_equal_objects(self):
        assert hash(Bits('101')) == hash(Bits('101'))

    def test_set_usage(self):
        s = {Bits('101'), Bits('101'), Bits('110')}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Bits - bitwise operators
# ---------------------------------------------------------------------------

class TestBitsOperators:
    def test_or(self):
        b = Bits('101') | Bits('011')
        assert isinstance(b, Bits)
        assert b.val == 7
        assert b.len == 3
        assert b.as_bin() == '111'

    def test_or_length_padding(self):
        b = Bits('1', 1) | Bits('1', 4)
        assert b.len == 4
        assert b.as_bin() == '0001'

    def test_or_non_bits(self):
        result = Bits('1').__or__(1)
        assert result is NotImplemented

    def test_or_with_non_bits_raises(self):
        with pytest.raises(TypeError):
            Bits('1') | 1

    def test_and(self):
        b = Bits('101') & Bits('011')
        assert isinstance(b, Bits)
        assert b.val == 1
        assert b.len == 3

    def test_and_length_padding(self):
        b = Bits('1', 1) & Bits('1', 4)
        assert b.len == 4
        assert b.as_bin() == '0001'

    def test_and_non_bits(self):
        result = Bits('1').__and__(1)
        assert result is NotImplemented

    def test_and_with_non_bits_raises(self):
        with pytest.raises(TypeError):
            Bits('1') & 1

    def test_lshift(self):
        b = Bits('1') << 2
        assert b.val == 4
        assert b.len == 3
        assert b.as_bin() == '100'

    def test_rshift(self):
        b = Bits('100') >> 2
        assert b.val == 1
        assert b.len == 1
        assert b.as_bin() == '1'


# ---------------------------------------------------------------------------
# Bits - conversions
# ---------------------------------------------------------------------------

class TestBitsConversions:
    def test_as_list(self):
        assert Bits('101').as_list() == [True, False, True]

    def test_as_list_empty(self):
        assert Bits('').as_list() == []

    def test_as_bin(self):
        assert Bits(5).as_bin() == '101'

    def test_as_bin_padded(self):
        assert Bits(5, 8).as_bin() == '00000101'

    def test_as_bin_empty(self):
        assert Bits('').as_bin() == ''

    def test_as_hex(self):
        assert Bits(255).as_hex() == 'FF'

    def test_as_hex_padded(self):
        assert Bits(10, 16).as_hex() == '000A'

    def test_as_hex_small(self):
        assert Bits(10).as_hex() == '0A'

    def test_as_hex_empty(self):
        assert Bits('').as_hex() == ''

    def test_as_hex_exact_byte(self):
        assert Bits(0xAB, 8).as_hex() == 'AB'

    def test_as_int(self):
        assert Bits('101').as_int() == 5

    def test_as_int_empty(self):
        assert Bits('').as_int() == 0

    def test_as_bytes(self):
        assert Bits('11111111').as_bytes() == b'\xff'

    def test_as_bytes_multi(self):
        assert Bits('0000000100000010').as_bytes() == b'\x01\x02'

    def test_as_bytes_even_pad(self):
        # 4 bits pads to one byte
        assert Bits('1010').as_bytes() == b'\x0a'

    def test_as_bytes_empty(self):
        assert Bits('').as_bytes() == b''


# ---------------------------------------------------------------------------
# Bits - classmethod constructors
# ---------------------------------------------------------------------------

class TestBitsConstructors:
    def test_from_list(self):
        assert Bits.from_list([True, False, True]) == Bits('101')

    def test_from_list_ints(self):
        assert Bits.from_list([1, 0, 1]) == Bits('101')

    def test_from_bin(self):
        assert Bits.from_bin('101') == Bits('101')

    def test_from_hex(self):
        assert Bits.from_hex('FF') == Bits(255, 8)

    def test_from_hex_with_prefix(self):
        assert Bits.from_hex('0xFF') == Bits(255, 8)

    def test_from_hex_empty(self):
        b = Bits.from_hex('')
        assert b.val == 0
        assert b.len == 0

    def test_from_hex_bytes(self):
        assert Bits.from_hex(b'FF') == Bits(255, 8)

    def test_from_hex_bytes_with_prefix(self):
        assert Bits.from_hex(b'0xFF') == Bits(255, 8)

    def test_from_int(self):
        assert Bits.from_int(5) == Bits('101')

    def test_from_int_with_len(self):
        b = Bits.from_int(5, 8)
        assert b.len == 8
        assert b.as_bin() == '00000101'

    def test_from_bytes(self):
        assert Bits.from_bytes(b'\x01\x02') == Bits('0000000100000010')

    def test_from_bytes_empty(self):
        b = Bits.from_bytes(b'')
        assert b.val == 0
        assert b.len == 0


# ---------------------------------------------------------------------------
# Bits - repr
# ---------------------------------------------------------------------------

class TestBitsRepr:
    def test_repr(self):
        assert repr(Bits('101')) == "Bits('101')"

    def test_repr_padded(self):
        assert repr(Bits(5, 8)) == "Bits('00000101')"

    def test_repr_empty(self):
        assert repr(Bits('')) == "Bits('')"
