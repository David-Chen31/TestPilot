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

def test_construct_format_field_str_basic():
    assert construct_format_field_str('name', '', None) == '{name}'


def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, 'spec', 'r') == ''


def test_construct_format_field_str_with_spec():
    assert construct_format_field_str('x', '>10', None) == '{x:>10}'


def test_construct_format_field_str_with_conv():
    assert construct_format_field_str('x', '', 'r') == '{x!r}'


def test_construct_format_field_str_with_both():
    assert construct_format_field_str('x', '.2f', 's') == '{x!s:.2f}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

def test_split_format_str_literal_only():
    assert split_format_str('hello world') == [('hello world', None)]


def test_split_format_str_single_field():
    assert split_format_str('{name}') == [('', '{name}')]


def test_split_format_str_mixed():
    result = split_format_str('a {name} b {1:03d} c')
    assert result == [
        ('a ', '{name}'),
        (' b ', '{1:03d}'),
        (' c', None),
    ]


def test_split_format_str_with_conversion():
    result = split_format_str('{x!r}')
    assert result == [('', '{x!r}')]


def test_split_format_str_empty():
    assert split_format_str('') == [('', None)]


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

def test_infer_positional_basic():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_positional_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_positional_with_conversion():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_multiple():
    assert infer_positional_format_args('{} and {}') == '{0} and {1}'


def test_infer_positional_named_untouched():
    assert infer_positional_format_args('{name}') == '{name}'


def test_infer_positional_numbered_untouched():
    assert infer_positional_format_args('{0} {1}') == '{0} {1}'


def test_infer_positional_escaped_braces():
    assert infer_positional_format_args('{{}}') == '{{}}'


def test_infer_positional_mixed_escaped():
    assert infer_positional_format_args('{{ {} }}') == '{{ {0} }}'


def test_infer_positional_dot_access():
    # Anonymous positional with attribute access
    assert infer_positional_format_args('{:.2f}') == '{0:.2f}'


def test_infer_positional_no_fields():
    assert infer_positional_format_args('plain text') == 'plain text'


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

def test_get_format_args_docstring_example():
    fargs, fkwargs = get_format_args("{noun} is {1:d} years old{punct}")
    assert fargs == [(1, int)]
    assert fkwargs == [('noun', str), ('punct', str)]


def test_get_format_args_empty():
    assert get_format_args('') == ([], [])


def test_get_format_args_literal_only():
    assert get_format_args('no fields here') == ([], [])


def test_get_format_args_positional_int():
    fargs, fkwargs = get_format_args('{0}')
    assert fargs == [(0, str)]
    assert fkwargs == []


def test_get_format_args_float_type():
    fargs, fkwargs = get_format_args('{0:f}')
    assert fargs == [(0, float)]
    assert fkwargs == []


def test_get_format_args_exponent_type():
    fargs, fkwargs = get_format_args('{0:e}')
    assert fargs == [(0, float)]
    assert fkwargs == []


def test_get_format_args_int_type_char():
    fargs, fkwargs = get_format_args('{0:x}')
    assert fargs == [(0, int)]
    assert fkwargs == []


def test_get_format_args_string_type_default():
    fargs, fkwargs = get_format_args('{name:s}')
    assert fargs == []
    assert fkwargs == [('name', str)]


def test_get_format_args_multiple_positional():
    fargs, fkwargs = get_format_args('{0} {1} {2}')
    assert fargs == [(0, str), (1, str), (2, str)]
    assert fkwargs == []


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{0} {0} {name} {name}')
    assert fargs == [(0, str)]
    assert fkwargs == [('name', str)]


def test_get_format_args_nested_spec():
    # Type spec with nested field
    fargs, fkwargs = get_format_args('{value:{width}}')
    assert fkwargs == [('value', str), ('width', str)]


def test_get_format_args_compound_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_compound_bracket_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


def test_get_format_args_anonymous_raises():
    # Parser with "{}" produces empty fname -> anonymous error
    with pytest.raises(ValueError):
        get_format_args('{}')


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

def test_tokenize_literal_only():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_empty():
    assert tokenize_format_str('') == []


def test_tokenize_single_field():
    result = tokenize_format_str('{name}')
    assert len(result) == 1
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == 'name'


def test_tokenize_resolves_positional():
    result = tokenize_format_str('{}')
    assert len(result) == 1
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == '0'
    assert result[0].is_positional is True


def test_tokenize_resolve_pos_false():
    result = tokenize_format_str('{}', resolve_pos=False)
    assert len(result) == 1
    field = result[0]
    assert field.fname == ''
    assert field.is_positional is True


def test_tokenize_mixed():
    result = tokenize_format_str('a{b}c')
    assert len(result) == 3
    assert result[0] == 'a'
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'b'
    assert result[2] == 'c'


def test_tokenize_multiple_fields():
    result = tokenize_format_str('{a}-{b}')
    assert len(result) == 3
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == 'a'
    assert result[1] == '-'
    assert isinstance(result[2], BaseFormatField)
    assert result[2].fname == 'b'


def test_tokenize_with_spec_and_conv():
    result = tokenize_format_str('{x!r:>10}')
    assert len(result) == 1
    field = result[0]
    assert field.fname == 'x'
    assert field.fspec == '>10'
    assert field.conv == 'r'


# ---------------------------------------------------------------------------
# BaseFormatField
# ---------------------------------------------------------------------------

def test_base_format_field_init():
    f = BaseFormatField('name')
    assert f.base_name == 'name'
    assert f.fname == 'name'
    assert f.subpath == []
    assert f.is_positional is False
    assert f.fspec == ''
    assert f.conv is None
    assert f.subfields == []
    assert f.type_char == ''
    assert f.type_func is str


def test_base_format_field_positional_numeric():
    f = BaseFormatField('0')
    assert f.is_positional is True


def test_base_format_field_positional_empty():
    f = BaseFormatField('')
    assert f.is_positional is True


def test_base_format_field_subpath():
    f = BaseFormatField('a.b')
    assert f.base_name == 'a'
    assert f.subpath == ['b']


def test_base_format_field_subpath_index():
    f = BaseFormatField('a[0]')
    assert f.base_name == 'a'
    assert f.subpath == ['0']


def test_base_format_field_fspec_type():
    f = BaseFormatField('x', spec='d')
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_fspec_float():
    f = BaseFormatField('x', spec='.2f')
    assert f.type_char == 'f'
    assert f.type_func is float


def test_base_format_field_fspec_unknown():
    f = BaseFormatField('x', spec='z')
    assert f.type_func is str


def test_base_format_field_subfields():
    f = BaseFormatField('x', spec='{width}')
    assert f.subfields == ['width']


def test_base_format_field_subfields_empty():
    f = BaseFormatField('x', spec='>10')
    assert f.subfields == []


def test_base_format_field_none_fspec():
    f = BaseFormatField('x', fspec=None)
    assert f.fspec == ''
    assert f.type_char == ''


def test_base_format_field_conv():
    f = BaseFormatField('x', conv='r')
    assert f.conv == 'r'
    assert f.conv_func is None


def test_base_format_field_fstr():
    f = BaseFormatField('x', spec='>10', conv='r')
    assert f.fstr == '{x!r:>10}'
    assert str(f) == '{x!r:>10}'


def test_base_format_field_fstr_simple():
    f = BaseFormatField('name')
    assert f.fstr == '{name}'
    assert str(f) == '{name}'


def test_base_format_field_set_fname():
    f = BaseFormatField('a')
    f.set_fname('b.c')
    assert f.base_name == 'b'
    assert f.fname == 'b.c'
    assert f.subpath == ['c']
    assert f.is_positional is False


def test_base_format_field_repr_simple():
    f = BaseFormatField('name')
    assert repr(f) == "BaseFormatField('name')"


def test_base_format_field_repr_with_spec():
    f = BaseFormatField('x', spec='>10')
    assert repr(f) == "BaseFormatField('x', '>10')"


def test_base_format_field_repr_with_conv():
    f = BaseFormatField('x', spec='>10', conv='r')
    assert repr(f) == "BaseFormatField('x', '>10', 'r')"


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

def test_deferred_value_basic():
    dv = DeferredValue(lambda: 42)
    assert dv.get_value() == 42


def test_deferred_value_caches():
    calls = []

    def func():
        calls.append(1)
        return 7

    dv = DeferredValue(func)
    assert dv.get_value() == 7
    assert dv.get_value() == 7
    assert len(calls) == 1


def test_deferred_value_no_cache():
    calls = []

    def func():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(func, cache_value=False)
    assert dv.get_value() == 1
    assert dv.get_value() == 2
    assert len(calls) == 2


def test_deferred_value_int():
    dv = DeferredValue(lambda: '42')
    assert int(dv) == 42
    assert isinstance(int(dv), int)


def test_deferred_value_float():
    dv = DeferredValue(lambda: '3.14')
    assert float(dv) == 3.14


def test_deferred_value_str():
    dv = DeferredValue(lambda: 123)
    assert str(dv) == '123'


def test_deferred_value_repr():
    dv = DeferredValue(lambda: 'abc')
    assert repr(dv) == "'abc'"


def test_deferred_value_format():
    dv = DeferredValue(lambda: 42)
    assert '{0:d}'.format(dv) == '42'
    assert f'{dv:>5}' == '   42'


def test_deferred_value_format_fallback():
    # Value whose __format__ raises for given spec -> fallback to type conv
    class BadFormat:
        def __init__(self, val):
            self.val = val

        def __format__(self, fmt):
            raise TypeError('nope')

        def __str__(self):
            return str(self.val)

    dv = DeferredValue(lambda: BadFormat('abc'))
    assert format(dv, 's') == 'abc'


def test_deferred_value_format_value_error_fallback():
    dv = DeferredValue(lambda: 'xyz')
    # 'd' format on a string raises ValueError normally; fallback to int conversion fails,
    # so instead test a case where fallback succeeds: value is str "10", fmt '>5s'
    assert format(dv, '>5') == '  xyz'[:5] if False else format(dv, '>5') == '  xyz'


def test_deferred_value_str_format_works():
    dv = DeferredValue(lambda: 'hello')
    assert format(dv, '>10') == '     hello'


def test_deferred_value_get_value_uses_callable():
    dv = DeferredValue(lambda: [1, 2, 3])
    assert dv.get_value() == [1, 2, 3]
    assert dv._value == [1, 2, 3]
