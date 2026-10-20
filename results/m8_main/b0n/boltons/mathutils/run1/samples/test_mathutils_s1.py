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
        assert clamp(-123, lower=0) == 0

    def test_defaults(self):
        assert clamp(42) == 42
        assert clamp(-1e10) == -1e10

    def test_equal_bounds(self):
        assert clamp(3, 3, 3) == 3

    def test_exact_boundary_values(self):
        assert clamp(0, 0, 5) == 0
        assert clamp(5, 0, 5) == 5

    def test_mixed_types(self):
        assert clamp(2.5, 0, 3) == 2.5
        assert clamp(2, 1.5, 3.5) == 2

    def test_bad_bounds_raises(self):
        with pytest.raises(ValueError):
            clamp(1, 5, 0)

    def test_bad_bounds_message(self):
        with pytest.raises(ValueError, match="upper bound"):
            clamp(1, 5, 0)


# ---------------------------------------------------------------------------
# ceil
# ---------------------------------------------------------------------------

VALID_CABLE_CSA = [1.5, 2.5, 4, 6, 10, 25, 35, 50]


class TestCeil:
    def test_plain(self):
        assert ceil(3.2) == 4
        assert ceil(3.0) == 3
        assert ceil(-1.2) == -1
        assert ceil(-2.0) == -2

    def test_plain_int(self):
        assert ceil(5) == 5

    def test_options_exact(self):
        assert ceil(4, options=VALID_CABLE_CSA) == 4

    def test_options_between(self):
        assert ceil(3.5, options=VALID_CABLE_CSA) == 4

    def test_options_unsorted(self):
        assert ceil(3.5, options=[50, 1.5, 25, 4]) == 4

    def test_options_first(self):
        assert ceil(1.0, options=VALID_CABLE_CSA) == 1.5

    def test_options_no_match_raises(self):
        with pytest.raises(ValueError):
            ceil(51, options=VALID_CABLE_CSA)

    def test_options_empty_raises(self):
        with pytest.raises(ValueError):
            ceil(1, options=[])


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

class TestFloor:
    def test_plain(self):
        assert floor(3.8) == 3
        assert floor(3.0) == 3
        assert floor(-1.2) == -2
        assert floor(-2.0) == -2

    def test_plain_int(self):
        assert floor(5) == 5

    def test_options_exact(self):
        assert floor(2.5, options=VALID_CABLE_CSA) == 2.5

    def test_options_between(self):
        assert floor(3.5, options=VALID_CABLE_CSA) == 2.5

    def test_options_unsorted(self):
        assert floor(3.5, options=[50, 1.5, 25, 4]) == 2.5

    def test_options_last(self):
        assert floor(60, options=VALID_CABLE_CSA) == 50

    def test_options_no_match_raises(self):
        with pytest.raises(ValueError):
            floor(1, options=VALID_CABLE_CSA)

    def test_options_empty_raises(self):
        with pytest.raises(ValueError):
            floor(1, options=[])


# ---------------------------------------------------------------------------
# Bits construction
# ---------------------------------------------------------------------------

class TestBitsConstruction:
    def test_from_int(self):
        b = Bits(5)
        assert b.val == 5
        assert b.len == 3

    def test_from_int_zero(self):
        b = Bits(0)
        assert b.val == 0
        assert b.len == 1

    def test_from_int_with_len(self):
        b = Bits(5, 8)
        assert b.val == 5
        assert b.len == 8

    def test_from_bin_string(self):
        b = Bits('101')
        assert b.val == 5
        assert b.len == 3

    def test_from_bin_string_leading_zeros(self):
        b = Bits('00101')
        assert b.val == 5
        assert b.len == 5

    def test_from_empty_string(self):
        b = Bits('')
        assert b.val == 0
        assert b.len == 0

    def test_from_hex_string(self):
        b = Bits('0xFF')
        assert b.val == 255
        assert b.len == 8

    def test_from_hex_string_lowercase(self):
        b = Bits('0xff')
        assert b.val == 255
        assert b.len == 8

    def test_from_bytes(self):
        b = Bits(b'101')
        assert b.val == 5
        assert b.len == 3

    def test_from_list(self):
        b = Bits([True, False, True])
        assert b.val == 5
        assert b.len == 3

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            Bits(-1)

    def test_too_large_for_len_raises(self):
        with pytest.raises(ValueError):
            Bits(8, 3)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            Bits(1.5)

    def test_bad_type_dict_raises(self):
        with pytest.raises(TypeError):
            Bits({'a': 1})


# ---------------------------------------------------------------------------
# Bits access
# ---------------------------------------------------------------------------

class TestBitsAccess:
    def setup_method(self):
        self.b = Bits('1011')  # val 11, len 4

    def test_len(self):
        assert len(self.b) == 4

    def test_getitem(self):
        assert self.b[0] is True
        assert self.b[1] is False
        assert self.b[2] is True
        assert self.b[3] is True

    def test_getitem_negative(self):
        assert self.b[-1] is True
        assert self.b[-2] is True
        assert self.b[-3] is False
        assert self.b[-4] is True

    def test_getitem_out_of_range(self):
        with pytest.raises(IndexError):
            self.b[4]
        with pytest.raises(IndexError):
            self.b[-5]

    def test_getitem_bad_type(self):
        with pytest.raises(TypeError):
            self.b['a']

    def test_getitem_slice(self):
        s = self.b[1:3]
        assert isinstance(s, Bits)
        assert s.as_bin() == '01'

    def test_slice_full(self):
        s = self.b[:]
        assert s.as_bin() == '1011'


# ---------------------------------------------------------------------------
# Bits equality / hash / operators
# ---------------------------------------------------------------------------

class TestBitsOperators:
    def test_eq(self):
        assert Bits(5, 3) == Bits('101')
        assert Bits(5) == Bits('101')

    def test_eq_different_len(self):
        assert Bits(5, 3) != Bits(5, 4)

    def test_eq_different_val(self):
        assert Bits(5) != Bits(6)

    def test_eq_other_type(self):
        assert Bits(5) != 5

    def test_hash(self):
        assert hash(Bits(5)) == hash(Bits(5))
        assert hash(Bits(5)) == hash(5)

    def test_or(self):
        r = Bits('1000') | Bits('0001')
        assert r.val == 9
        assert r.len == 4

    def test_or_different_len(self):
        r = Bits('10') | Bits('0100')  # val 2 len 2, val 4 len 4
        assert r.val == 6
        assert r.len == 4

    def test_or_other_type(self):
        with pytest.raises(TypeError):
            Bits(1) | 1

    def test_and(self):
        r = Bits('1010') & Bits('0110')
        assert r.val == 0b0010
        assert r.len == 4

    def test_and_different_len(self):
        r = Bits('10') & Bits('0110')  # val 2 len 2 & val 6 len 4
        assert r.val == 2
        assert r.len == 4

    def test_and_other_type(self):
        with pytest.raises(TypeError):
            Bits(1) & 1

    def test_lshift(self):
        r = Bits('101') << 2
        assert r.val == 0b10100
        assert r.len == 5

    def test_rshift(self):
        r = Bits('10100') >> 2
        assert r.val == 0b101
        assert r.len == 3


# ---------------------------------------------------------------------------
# Bits conversions
# ---------------------------------------------------------------------------

class TestBitsConversions:
    def test_as_int(self):
        assert Bits('1011').as_int() == 11

    def test_as_bin(self):
        assert Bits(5, 8).as_bin() == '00000101'

    def test_as_bin_empty(self):
        assert Bits('').as_bin() == ''

    def test_as_list(self):
        assert Bits('1011').as_list() == [True, False, True, True]

    def test_as_list_empty(self):
        assert Bits('').as_list() == []

    def test_as_hex(self):
        assert Bits('0x0F').as_hex() == '0F'

    def test_as_hex_padding(self):
        assert Bits(5, 8).as_hex() == '05'

    def test_as_hex_empty(self):
        assert Bits('').as_hex() == ''

    def test_as_bytes(self):
        assert Bits('0xFF').as_bytes() == b'\xff'

    def test_as_bytes_padded(self):
        assert Bits(5, 8).as_bytes() == b'\x05'

    def test_repr(self):
        assert repr(Bits('101')) == "Bits('101')"

    def test_repr_zero(self):
        assert repr(Bits(0)) == "Bits('0')"


# ---------------------------------------------------------------------------
# Bits classmethods
# ---------------------------------------------------------------------------

class TestBitsClassmethods:
    def test_from_list(self):
        b = Bits.from_list([True, False, True])
        assert b.val == 5
        assert b.len == 3

    def test_from_bin(self):
        b = Bits.from_bin('1011')
        assert b.val == 11
        assert b.len == 4

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
        b = Bits.from_bytes(b'\xff')
        assert b.val == 255
        assert b.len == 8

    def test_from_bytes_multi(self):
        b = Bits.from_bytes(b'\x01\x02')
        assert b.val == 0x0102
        assert b.len == 16


# ---------------------------------------------------------------------------
# Bits immutability (slots-based)
# ---------------------------------------------------------------------------

class TestBitsSlots:
    def test_no_arbitrary_attributes(self):
        b = Bits(5)
        with pytest.raises(AttributeError):
            b.foo = 1

    def test_no_dict(self):
        b = Bits(5)
        assert not hasattr(b, '__dict__')
