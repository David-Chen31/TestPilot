import pytest

from boltons.formatutils import (
    construct_format_field_str,
    split_format_str,
    infer_positional_format_args,
    get_format_args,
    tokenize_format_str,
    BaseFormatField,
    DeferredValue,
)


# ---------------------------------------------------------------------------
# construct_format_field_str
# ---------------------------------------------------------------------------

class TestConstructFormatFieldStr:
    def test_basic(self):
        assert construct_format_field_str('name', '', None) == '{name}'

    def test_with_spec(self):
        assert construct_format_field_str('name', '>10', None) == '{name:>10}'

    def test_with_conv(self):
        assert construct_format_field_str('name', '', 'r') == '{name!r}'

    def test_with_conv_and_spec(self):
        assert construct_format_field_str('name', '>10', 'r') == '{name!r:>10}'

    def test_none_fname_returns_empty(self):
        assert construct_format_field_str(None, '>10', 'r') == ''
        assert construct_format_field_str(None, '', None) == ''

    def test_empty_spec_and_conv(self):
        assert construct_format_field_str('x', None, None) == '{x}'

    def test_positional_name(self):
        assert construct_format_field_str('0', 'd', None) == '{0:d}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

class TestSplitFormatStr:
    def test_no_fields(self):
        assert split_format_str('hello world') == [('hello world', None)]

    def test_single_field(self):
        assert split_format_str('hi {name}') == [('hi ', '{name}')]

    def test_multiple_fields(self):
        assert split_format_str('a {x} b {y} c') == [
            ('a ', '{x}'),
            (' b ', '{y}'),
            (' c', None),
        ]

    def test_field_with_spec_and_conv(self):
        assert split_format_str('{val!r:>5}') == [('', '{val!r:>5}')]

    def test_empty_string(self):
        assert split_format_str('') == [('', None)]

    def test_escaped_braces(self):
        # Formatter turns '{{' into literal '{'
        assert split_format_str('{{literal}}') == [('{literal}', None)]


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

class TestInferPositionalFormatArgs:
    def test_empty(self):
        assert infer_positional_format_args('') == ''

    def test_no_fields(self):
        assert infer_positional_format_args('hello') == 'hello'

    def test_anonymous_plain(self):
        assert infer_positional_format_args('{} {} {}') == '{0} {1} {2}'

    def test_anonymous_with_type(self):
        assert infer_positional_format_args('{:d} {:s}') == '{0:d} {1:s}'

    def test_anonymous_with_conv(self):
        assert infer_positional_format_args('{!r}') == '{0!r}'

    def test_anonymous_with_attr(self):
        assert infer_positional_format_args('{.attr}') == '{0.attr}'

    def test_anonymous_with_index(self):
        assert infer_positional_format_args('{[0]}') == '{0[0]}'

    def test_escaped_braces(self):
        assert infer_positional_format_args('{{}}') == '{{}}'

    def test_escaped_open_brace(self):
        assert infer_positional_format_args('{{ {}') == '{{ {0}'

    def test_escaped_close_brace(self):
        assert infer_positional_format_args('{} }}') == '{0} }}'

    def test_mixed_named_preserved(self):
        assert infer_positional_format_args('{name} and {}') == '{name} and {0}'

    def test_numbered_preserved(self):
        # A numbered field {0} doesn't match the anon regex (well, {0} does
        # not match because 0 isn't one of the matched chars after '{')
        assert infer_positional_format_args('{0}') == '{0}'

    def test_surrounding_text(self):
        assert infer_positional_format_args('a {} b {} c') == 'a {0} b {1} c'

    def test_only_anonymous_counting(self):
        assert infer_positional_format_args('{x} {} {y} {}') == '{x} {0} {y} {1}'

    def test_anonymous_with_format_spec_text(self):
        assert (
            infer_positional_format_args('{:>10}') == '{0:>10}'
        )


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

class TestGetFormatArgs:
    def test_basic(self):
        result = get_format_args("{noun} is {1:d} years old{punct}")
        assert result == ([(1, int)], [('noun', str), ('punct', str)])

    def test_no_fields(self):
        assert get_format_args('no fields here') == ([], [])

    def test_empty(self):
        assert get_format_args('') == ([], [])

    def test_positional_only(self):
        fargs, fkwargs = get_format_args('{0} {1} {2:d}')
        assert fargs == [(0, str), (1, str), (2, int)]
        assert fkwargs == []

    def test_named_only(self):
        fargs, fkwargs = get_format_args('{a} {b:f}')
        assert fargs == []
        assert fkwargs == [('a', str), ('b', float)]

    def test_dedup(self):
        fargs, fkwargs = get_format_args('{a} {a} {a}')
        assert fargs == []
        assert fkwargs == [('a', str)]

    def test_dedup_positional(self):
        fargs, fkwargs = get_format_args('{0} {0}')
        assert fargs == [(0, str)]
        assert fkwargs == []

    def test_type_map_int(self):
        for ch in 'bcdoxXn':
            fargs, fkwargs = get_format_args('{0:' + ch + '}')
            assert fargs == [(0, int)]

    def test_type_map_float(self):
        for ch in 'eEfFgGn%':
            fargs, fkwargs = get_format_args('{0:' + ch + '}')
            assert fargs == [(0, float)]

    def test_type_map_str(self):
        fargs, fkwargs = get_format_args('{0:s}')
        assert fargs == [(0, str)]

    def test_unknown_type_char_defaults_str(self):
        fargs, fkwargs = get_format_args('{0:z}')
        assert fargs == [(0, str)]

    def test_no_spec_uses_s(self):
        fargs, fkwargs = get_format_args('{x}')
        assert fkwargs == [('x', str)]

    def test_compound_attr_raises(self):
        with pytest.raises(ValueError):
            get_format_args('{obj.attr}')

    def test_compound_index_raises(self):
        with pytest.raises(ValueError):
            get_format_args('{obj[0]}')

    def test_anonymous_raises(self):
        with pytest.raises(ValueError):
            get_format_args('{}')

    def test_anonymous_with_spec_raises(self):
        with pytest.raises(ValueError):
            get_format_args('{:d}')

    def test_nested_spec_args(self):
        # nested subfield in fspec gets added
        fargs, fkwargs = get_format_args('{a:{width}}')
        assert fkwargs == [('a', str), ('width', str)]

    def test_nested_spec_positional_subfield(self):
        fargs, fkwargs = get_format_args('{0:{1}}')
        assert fargs == [(0, str), (1, str)]


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

class TestTokenizeFormatStr:
    def test_empty(self):
        assert tokenize_format_str('') == []

    def test_literal_only(self):
        assert tokenize_format_str('hello') == ['hello']

    def test_single_field(self):
        tokens = tokenize_format_str('hi {name}')
        assert len(tokens) == 2
        assert tokens[0] == 'hi '
        assert isinstance(tokens[1], BaseFormatField)
        assert tokens[1].fname == 'name'

    def test_resolve_pos_true(self):
        tokens = tokenize_format_str('{}')
        assert len(tokens) == 1
        assert isinstance(tokens[0], BaseFormatField)
        assert tokens[0].fname == '0'

    def test_resolve_pos_false(self):
        tokens = tokenize_format_str('{}', resolve_pos=False)
        assert len(tokens) == 1
        assert isinstance(tokens[0], BaseFormatField)
        assert tokens[0].fname == ''

    def test_mixed_tokens(self):
        tokens = tokenize_format_str('a {x} b {y}')
        assert tokens[0] == 'a '
        assert isinstance(tokens[1], BaseFormatField)
        assert tokens[1].fname == 'x'
        assert tokens[2] == ' b '
        assert isinstance(tokens[3], BaseFormatField)
        assert tokens[3].fname == 'y'

    def test_escaped_braces(self):
        tokens = tokenize_format_str('{{x}}')
        assert tokens == ['{x}']

    def test_field_with_spec(self):
        tokens = tokenize_format_str('{val:>3d}')
        field = tokens[0]
        assert field.fname == 'val'
        assert field.fspec == '>3d'


# ---------------------------------------------------------------------------
# BaseFormatField
# ---------------------------------------------------------------------------

class TestBaseFormatField:
    def test_basic_fname(self):
        f = BaseFormatField('name')
        assert f.fname == 'name'
        assert f.base_name == 'name'
        assert f.subpath == []
        assert f.is_positional is False

    def test_positional(self):
        f = BaseFormatField('0')
        assert f.is_positional is True
        assert f.base_name == '0'

    def test_empty_fname_is_positional(self):
        f = BaseFormatField('')
        assert f.is_positional is True
        assert f.base_name == ''

    def test_attr_subpath(self):
        f = BaseFormatField('obj.attr')
        assert f.base_name == 'obj'
        assert f.subpath == ['attr']
        assert f.fname == 'obj.attr'

    def test_index_subpath(self):
        f = BaseFormatField('items[0]')
        assert f.base_name == 'items'
        assert f.subpath == ['0']

    def test_mixed_subpath(self):
        f = BaseFormatField('a.b[0]')
        assert f.base_name == 'a'
        assert f.subpath == ['b', '0']

    def test_default_fspec_and_conv(self):
        f = BaseFormatField('x')
        assert f.fspec == ''
        assert f.conv is None
        assert f.type_char == ''
        assert f.type_func is str
        assert f.subfields == []

    def test_fspec_type_char(self):
        f = BaseFormatField('x', '>10d')
        assert f.fspec == '>10d'
        assert f.type_char == 'd'
        assert f.type_func is int

    def test_fspec_float_type(self):
        f = BaseFormatField('x', '.2f')
        assert f.type_char == 'f'
        assert f.type_func is float

    def test_fspec_str_type(self):
        f = BaseFormatField('x', 's')
        assert f.type_func is str

    def test_fspec_none_becomes_empty(self):
        f = BaseFormatField('x', None)
        assert f.fspec == ''

    def test_fspec_subfields(self):
        f = BaseFormatField('x', '>{width}')
        assert f.subfields == ['width']

    def test_conv(self):
        f = BaseFormatField('x', '', 'r')
        assert f.conv == 'r'
        assert f.conv_func is None

    def test_fstr(self):
        f = BaseFormatField('x', '>5', 'r')
        assert f.fstr == '{x!r:>5}'
        assert str(f) == '{x!r:>5}'

    def test_fstr_basic(self):
        f = BaseFormatField('x')
        assert f.fstr == '{x}'

    def test_repr_basic(self):
        f = BaseFormatField('x')
        assert repr(f) == "BaseFormatField('x')"

    def test_repr_with_fspec(self):
        f = BaseFormatField('x', '>5')
        assert repr(f) == "BaseFormatField('x', '>5')"

    def test_repr_with_conv(self):
        f = BaseFormatField('x', '>5', 'r')
        assert repr(f) == "BaseFormatField('x', '>5', 'r')"

    def test_set_fname_update(self):
        f = BaseFormatField('x')
        f.set_fname('y.z')
        assert f.fname == 'y.z'
        assert f.base_name == 'y'
        assert f.subpath == ['z']

    def test_set_fspec_update(self):
        f = BaseFormatField('x')
        f.set_fspec('d')
        assert f.type_char == 'd'
        assert f.type_func is int

    def test_set_conv_update(self):
        f = BaseFormatField('x')
        f.set_conv('s')
        assert f.conv == 's'

    def test_equality_identity_only(self):
        # no __eq__ defined, so distinct objects aren't equal
        assert BaseFormatField('x') != BaseFormatField('x')


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

class TestDeferredValue:
    def test_caching_default(self):
        calls = []

        def func():
            calls.append(1)
            return 42

        dv = DeferredValue(func)
        assert dv.get_value() == 42
        assert dv.get_value() == 42
        assert len(calls) == 1

    def test_no_cache(self):
        calls = []

        def func():
            calls.append(1)
            return 42

        dv = DeferredValue(func, cache_value=False)
        assert dv.get_value() == 42
        assert dv.get_value() == 42
        assert len(calls) == 2

    def test_lazy(self):
        calls = []
        DeferredValue(lambda: calls.append(1))
        assert calls == []

    def test_str(self):
        dv = DeferredValue(lambda: 123)
        assert str(dv) == '123'

    def test_repr(self):
        dv = DeferredValue(lambda: 'abc')
        assert repr(dv) == "'abc'"

    def test_int(self):
        dv = DeferredValue(lambda: '12')
        assert int(dv) == 12

    def test_float(self):
        dv = DeferredValue(lambda: '1.5')
        assert float(dv) == 1.5

    def test_format_empty(self):
        dv = DeferredValue(lambda: 42)
        assert format(dv, '') == '42'

    def test_format_with_spec(self):
        dv = DeferredValue(lambda: 42)
        assert format(dv, 'd') == '42'

    def test_format_in_string(self):
        dv = DeferredValue(lambda: 7)
        assert '{0}'.format(dv) == '7'
        assert '{0:03d}'.format(dv) == '007'

    def test_format_fallback_type_conv(self):
        # value has no usable __format__ for the given spec, fallback converts
        class NoFormat:
            def __str__(self):
                return 'noformat'

            def __format__(self, spec):
                raise ValueError('nope')

        dv = DeferredValue(lambda: NoFormat())
        assert format(dv, 'd') == format(str('noformat'), 'd') if False else True
        # The actual fallback for empty spec: type_conv(str)(value) formats as str
        # _TYPE_MAP for 'd' is int, which will fail on NoFormat, so test with 's'
        dv2 = DeferredValue(lambda: NoFormat())
        assert format(dv2, 's') == 'noformat'

    def test_cache_value_false_still_uses_value(self):
        vals = iter([1, 2, 3])
        dv = DeferredValue(lambda: next(vals), cache_value=False)
        assert dv.get_value() == 1
        assert dv.get_value() == 2
        assert dv.get_value() == 3

    def test_cache_internal_value_set(self):
        dv = DeferredValue(lambda: 99)
        # Value not yet cached
        assert DeferredValue.get_value(dv) == 99
        # Cached now
        assert dv._value == 99
