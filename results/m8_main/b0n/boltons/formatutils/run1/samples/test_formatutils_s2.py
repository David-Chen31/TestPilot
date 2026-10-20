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


# construct_format_field_str

def test_construct_format_field_str_basic():
    assert construct_format_field_str('name', '', None) == '{name}'


def test_construct_format_field_str_with_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'


def test_construct_format_field_str_with_spec():
    assert construct_format_field_str('name', '>10', None) == '{name:>10}'


def test_construct_format_field_str_with_conv_and_spec():
    assert construct_format_field_str('value', '.2f', 'r') == '{value!r:.2f}'


def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, '>10', 'r') == ''


def test_construct_format_field_str_empty_fname():
    # base_name empty -> treated like normal field
    assert construct_format_field_str('', '', None) == '{}'


# split_format_str

def test_split_format_str_literal_only():
    assert split_format_str('hello world') == [('hello world', None)]


def test_split_format_str_simple_field():
    assert split_format_str('a {x} b') == [('a ', '{x}'), (' b', None)]


def test_split_format_str_multiple_fields():
    result = split_format_str('{a}-{b}')
    assert result == [('', '{a}'), ('-', '{b}')]


def test_split_format_str_with_spec_and_conv():
    result = split_format_str('{x!r:>10}')
    assert result == [('', '{x!r:>10}')]


def test_split_format_str_escaped_braces():
    # Escaped braces are literals; parse yields them in the lit.
    result = split_format_str('{{literal}}')
    assert result == [('{{literal}}', None)]


# infer_positional_format_args

def test_infer_positional_basic():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_positional_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_positional_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_with_dot():
    # "{.2f}" is an anon positional with . spec start
    assert infer_positional_format_args('{.2f}') == '{0.2f}'


def test_infer_positional_with_bracket():
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


def test_infer_positional_multiple():
    assert infer_positional_format_args('{} and {}') == '{0} and {1}'


def test_infer_positional_mixed_numbered():
    # numbered args are left alone
    assert infer_positional_format_args('{0} and {}') == '{0} and {1}'


def test_infer_positional_escaped_braces():
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_positional_no_fields():
    assert infer_positional_format_args('no fields here') == 'no fields here'


def test_infer_positional_literal_before():
    assert infer_positional_format_args('prefix {} suffix') == 'prefix {0} suffix'


# get_format_args

def test_get_format_args_docstring_example():
    fargs, fkwargs = get_format_args("{noun} is {1:d} years old{punct}")
    assert fargs == [(1, int)]
    assert fkwargs == [('noun', str), ('punct', str)]


def test_get_format_args_empty():
    assert get_format_args('') == ([], [])


def test_get_format_args_only_literal():
    assert get_format_args('hello') == ([], [])


def test_get_format_args_positional_int():
    fargs, fkwargs = get_format_args('{0:d}')
    assert fargs == [(0, int)]
    assert fkwargs == []


def test_get_format_args_positional_float():
    fargs, fkwargs = get_format_args('{0:f}')
    assert fargs == [(0, float)]
    assert fkwargs == []


def test_get_format_args_string_type():
    fargs, fkwargs = get_format_args('{0:s}')
    assert fargs == [(0, str)]
    assert fkwargs == []


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{a} {a} {a}')
    assert fkwargs == [('a', str)]
    assert fargs == []


def test_get_format_args_multiple_kwargs():
    fargs, fkwargs = get_format_args('{a} {b}')
    assert fkwargs == [('a', str), ('b', str)]


def test_get_format_args_anon_positional_raises():
    with pytest.raises(ValueError):
        get_format_args('{}')


def test_get_format_args_compound_format_arg_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_index_compound_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


def test_get_format_args_nested_subfield():
    # subfield in spec gets added with default str
    fargs, fkwargs = get_format_args('{a:{width}}')
    assert ('a', str) in fkwargs
    assert ('width', str) in fkwargs


def test_get_format_args_conv():
    fargs, fkwargs = get_format_args('{a!r}')
    assert fkwargs == [('a', str)]


# tokenize_format_str

def test_tokenize_format_str_literal_only():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_format_str_empty():
    assert tokenize_format_str('') == []


def test_tokenize_format_str_simple_field():
    result = tokenize_format_str('hi {name}')
    assert result == ['hi ', BaseFormatField('name')]


def test_tokenize_format_str_resolve_pos():
    result = tokenize_format_str('{}')
    assert len(result) == 1
    field = result[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == '0'


def test_tokenize_format_str_no_resolve_pos():
    result = tokenize_format_str('{}', resolve_pos=False)
    assert len(result) == 1
    field = result[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == ''


def test_tokenize_format_str_multiple_fields():
    result = tokenize_format_str('a{x}b{y}c')
    assert len(result) == 5
    assert result[0] == 'a'
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'x'
    assert result[2] == 'b'
    assert result[3].fname == 'y'
    assert result[4] == 'c'


def test_tokenize_format_str_escaped_braces():
    result = tokenize_format_str('{{x}}')
    assert result == ['{{x}}']


def test_tokenize_format_str_with_spec_conv():
    result = tokenize_format_str('{x!r:>10}')
    field = result[0]
    assert field.fname == 'x'
    assert field.fspec == '>10'
    assert field.conv == 'r'


# BaseFormatField

def test_base_format_field_init():
    f = BaseFormatField('name')
    assert f.fname == 'name'
    assert f.base_name == 'name'
    assert f.subpath == []
    assert f.is_positional is False
    assert f.fspec == ''
    assert f.conv is None


def test_base_format_field_positional_digit():
    f = BaseFormatField('0')
    assert f.is_positional is True


def test_base_format_field_positional_empty():
    f = BaseFormatField('')
    assert f.is_positional is True


def test_base_format_field_subpath_dot():
    f = BaseFormatField('a.b')
    assert f.base_name == 'a'
    assert f.subpath == ['b']


def test_base_format_field_subpath_bracket():
    f = BaseFormatField('a[0]')
    assert f.base_name == 'a'
    assert f.subpath == ['0']


def test_base_format_field_type_char_int():
    f = BaseFormatField('x', '>10d')
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_type_char_float():
    f = BaseFormatField('x', '>10f')
    assert f.type_char == 'f'
    assert f.type_func is float


def test_base_format_field_type_char_str():
    f = BaseFormatField('x', '>10s')
    assert f.type_char == 's'
    assert f.type_func is str


def test_base_format_field_default_type():
    f = BaseFormatField('x', '')
    assert f.type_char == ''
    assert f.type_func is str


def test_base_format_field_subfields():
    f = BaseFormatField('x', '{width}')
    assert f.subfields == ['width']


def test_base_format_field_no_subfields():
    f = BaseFormatField('x', '>10')
    assert f.subfields == []


def test_base_format_field_conv():
    f = BaseFormatField('x', '', 'r')
    assert f.conv == 'r'
    assert f.conv_func is None


def test_base_format_field_fstr_property():
    f = BaseFormatField('x', '>10', 'r')
    assert f.fstr == '{x!r:>10}'


def test_base_format_field_str():
    f = BaseFormatField('x', '>10', 'r')
    assert str(f) == '{x!r:>10}'


def test_base_format_field_repr_plain():
    f = BaseFormatField('x')
    assert repr(f) == "BaseFormatField('x')"


def test_base_format_field_repr_spec():
    f = BaseFormatField('x', '>10')
    assert repr(f) == "BaseFormatField('x', '>10')"


def test_base_format_field_repr_conv():
    f = BaseFormatField('x', '>10', 'r')
    assert repr(f) == "BaseFormatField('x', '>10', 'r')"


def test_base_format_field_set_fname_update():
    f = BaseFormatField('x')
    f.set_fname('0')
    assert f.is_positional is True
    assert f.fname == '0'


def test_base_format_field_set_fspec_update():
    f = BaseFormatField('x')
    f.set_fspec('>10d')
    assert f.type_func is int
    assert f.fspec == '>10d'


def test_base_format_field_set_conv_update():
    f = BaseFormatField('x')
    f.set_conv('r')
    assert f.conv == 'r'


# DeferredValue

def test_deferred_value_basic():
    dv = DeferredValue(lambda: 42)
    assert int(dv) == 42


def test_deferred_value_cache():
    calls = []

    def func():
        calls.append(1)
        return 5

    dv = DeferredValue(func)
    assert int(dv) == 5
    assert int(dv) == 5
    assert len(calls) == 1


def test_deferred_value_no_cache():
    calls = []

    def func():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(func, cache_value=False)
    assert int(dv) == 1
    assert int(dv) == 2


def test_deferred_value_float():
    dv = DeferredValue(lambda: '3.5')
    assert float(dv) == 3.5


def test_deferred_value_str():
    dv = DeferredValue(lambda: 123)
    assert str(dv) == '123'


def test_deferred_value_repr():
    dv = DeferredValue(lambda: 'abc')
    assert repr(dv) == "'abc'"


def test_deferred_value_unicode():
    dv = DeferredValue(lambda: 7)
    assert dv.__unicode__() == '7'


def test_deferred_value_format():
    dv = DeferredValue(lambda: 3.14159)
    assert format(dv, '.2f') == '3.14'


def test_deferred_value_format_in_str():
    dv = DeferredValue(lambda: 42)
    assert "value={:d}".format(dv) == "value=42"


def test_deferred_value_format_fallback():
    # value that fails its own __format__ falls back to type_conv
    class Weird:
        def __format__(self, fmt):
            raise ValueError("nope")

        def __str__(self):
            return "weird"

    dv = DeferredValue(lambda: Weird())
    assert format(dv, 's') == 'weird'


def test_deferred_value_get_value():
    dv = DeferredValue(lambda: 10)
    assert dv.get_value() == 10


def test_deferred_value_get_value_cached():
    calls = []

    def func():
        calls.append(1)
        return 99

    dv = DeferredValue(func)
    assert dv.get_value() == 99
    assert dv.get_value() == 99
    assert len(calls) == 1


def test_deferred_value_get_value_no_cache():
    calls = []

    def func():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(func, cache_value=False)
    assert dv.get_value() == 1
    assert dv.get_value() == 2
