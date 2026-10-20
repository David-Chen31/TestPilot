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


def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, '', None) == ''
    assert construct_format_field_str(None, 'd', 'r') == ''


def test_construct_format_field_str_with_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'
    assert construct_format_field_str('name', '', 's') == '{name!s}'


def test_construct_format_field_str_with_fspec():
    assert construct_format_field_str('name', '>10', None) == '{name:>10}'


def test_construct_format_field_str_all():
    assert construct_format_field_str('name', '>10', 'r') == '{name!r:>10}'


# split_format_str
def test_split_format_str_plain():
    assert split_format_str('hello') == [('hello', None)]


def test_split_format_str_empty():
    assert split_format_str('') == [('', None)]


def test_split_format_str_single_field():
    assert split_format_str('{name}') == [('', '{name}')]


def test_split_format_str_mixed():
    result = split_format_str('a{name}b{age:d}c')
    assert result == [('a', '{name}'), ('b', '{age:d}'), ('c', None)]


def test_split_format_str_with_conv_and_spec():
    result = split_format_str('{x!r:>5}')
    assert result == [('', '{x!r:>5}')]


# infer_positional_format_args
def test_infer_positional_no_anon():
    assert infer_positional_format_args('{name}') == '{name}'
    assert infer_positional_format_args('{0} {1}') == '{0} {1}'


def test_infer_positional_empty():
    assert infer_positional_format_args('') == ''


def test_infer_positional_literal_only():
    assert infer_positional_format_args('hello world') == 'hello world'


def test_infer_positional_single_anon():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_positional_multiple_anon():
    assert infer_positional_format_args('{} and {}') == '{0} and {1}'


def test_infer_positional_anon_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_anon_mixed():
    assert infer_positional_format_args('a {} b {name} c {}') == 'a {0} b {name} c {1}'


def test_infer_positional_escaped_braces():
    assert infer_positional_format_args('{{}}') == '{{}}'
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_positional_anon_with_attr():
    assert infer_positional_format_args('{.foo}') == '{0.foo}'
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


# get_format_args
def test_get_format_args_docstring_example():
    assert get_format_args("{noun} is {1:d} years old{punct}") == (
        [(1, int)],
        [('noun', str), ('punct', str)],
    )


def test_get_format_args_empty():
    assert get_format_args('') == ([], [])


def test_get_format_args_literal_only():
    assert get_format_args('no fields here') == ([], [])


def test_get_format_args_positional_types():
    fargs, fkwargs = get_format_args('{0:d}{1:f}{2:s}')
    assert fargs == [(0, int), (1, float), (2, str)]
    assert fkwargs == []


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{a}{a}{b}')
    assert fargs == []
    assert fkwargs == [('a', str), ('b', str)]


def test_get_format_args_compound_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError):
        get_format_args('{}')


def test_get_format_args_nested_spec():
    fargs, fkwargs = get_format_args('{value:{width}}')
    assert fargs == []
    assert fkwargs == [('value', str), ('width', str)]


def test_get_format_args_float_type():
    fargs, fkwargs = get_format_args('{x:%}')
    assert fkwargs == [('x', float)]


# tokenize_format_str
def test_tokenize_format_str_plain():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_format_str_empty():
    assert tokenize_format_str('') == []


def test_tokenize_format_str_single_field():
    result = tokenize_format_str('{name}')
    assert len(result) == 1
    field = result[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == 'name'


def test_tokenize_format_str_mixed():
    result = tokenize_format_str('a{name}b')
    assert len(result) == 3
    assert result[0] == 'a'
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'name'
    assert result[2] == 'b'


def test_tokenize_format_str_resolve_pos():
    result = tokenize_format_str('{}')
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == '0'
    assert result[0].is_positional


def test_tokenize_format_str_no_resolve_pos():
    result = tokenize_format_str('{}', resolve_pos=False)
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == ''


def test_tokenize_format_str_literal_dropped_when_falsy():
    result = tokenize_format_str('{a}{b}')
    assert len(result) == 2
    assert all(isinstance(f, BaseFormatField) for f in result)


# BaseFormatField
def test_base_format_field_basic():
    field = BaseFormatField('name')
    assert field.fname == 'name'
    assert field.base_name == 'name'
    assert field.subpath == []
    assert field.fspec == ''
    assert field.conv is None
    assert field.is_positional is False


def test_base_format_field_positional_digit():
    field = BaseFormatField('0')
    assert field.is_positional is True
    assert field.base_name == '0'


def test_base_format_field_positional_empty():
    field = BaseFormatField('')
    assert field.is_positional is True
    assert field.base_name == ''


def test_base_format_field_subpath():
    field = BaseFormatField('a.b')
    assert field.base_name == 'a'
    assert field.subpath == ['b']


def test_base_format_field_index_subpath():
    field = BaseFormatField('a[0]')
    assert field.base_name == 'a'
    assert field.subpath == ['0]']


def test_base_format_field_fspec_type_char():
    field = BaseFormatField('x', 'd')
    assert field.type_char == 'd'
    assert field.type_func is int


def test_base_format_field_fspec_float():
    field = BaseFormatField('x', 'f')
    assert field.type_func is float


def test_base_format_field_fspec_str():
    field = BaseFormatField('x', 's')
    assert field.type_func is str


def test_base_format_field_fspec_unknown_char():
    field = BaseFormatField('x', 'z')
    assert field.type_func is str


def test_base_format_field_fspec_none():
    field = BaseFormatField('x', None)
    assert field.fspec == ''


def test_base_format_field_subfields():
    field = BaseFormatField('x', '{width}')
    assert field.subfields == ['width']


def test_base_format_field_no_subfields():
    field = BaseFormatField('x', '>10')
    assert field.subfields == []


def test_base_format_field_conv():
    field = BaseFormatField('x', '', 'r')
    assert field.conv == 'r'
    assert field.conv_func is None


def test_base_format_field_fstr():
    field = BaseFormatField('name', '>10', 'r')
    assert field.fstr == '{name!r:>10}'


def test_base_format_field_str():
    field = BaseFormatField('name', '>10', 'r')
    assert str(field) == '{name!r:>10}'


def test_base_format_field_repr_basic():
    field = BaseFormatField('name')
    assert repr(field) == "BaseFormatField('name')"


def test_base_format_field_repr_with_spec():
    field = BaseFormatField('name', '>10')
    assert repr(field) == "BaseFormatField('name', '>10')"


def test_base_format_field_repr_with_conv():
    field = BaseFormatField('name', '>10', 'r')
    assert repr(field) == "BaseFormatField('name', '>10', 'r')"


# DeferredValue
def test_deferred_value_caches_by_default():
    calls = []

    def compute():
        calls.append(1)
        return 42

    dv = DeferredValue(compute)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
    assert len(calls) == 1


def test_deferred_value_no_cache():
    calls = []

    def compute():
        calls.append(1)
        return 42

    dv = DeferredValue(compute, cache_value=False)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
    assert len(calls) == 2


def test_deferred_value_int():
    dv = DeferredValue(lambda: '42')
    assert int(dv) == 42


def test_deferred_value_float():
    dv = DeferredValue(lambda: '1.5')
    assert float(dv) == 1.5


def test_deferred_value_str():
    dv = DeferredValue(lambda: 123)
    assert str(dv) == '123'


def test_deferred_value_repr():
    dv = DeferredValue(lambda: 123)
    assert repr(dv) == '123'


def test_deferred_value_format_basic():
    dv = DeferredValue(lambda: 42)
    assert '{0}'.format(dv) == '42'


def test_deferred_value_format_spec():
    dv = DeferredValue(lambda: 42)
    assert '{0:>5}'.format(dv) == '   42'


def test_deferred_value_format_fallback():
    dv = DeferredValue(lambda: '42')
    # string doesn't accept 'd', fallback converts via _TYPE_MAP to int
    assert '{0:d}'.format(dv) == '42'


def test_deferred_value_format_float_fallback():
    dv = DeferredValue(lambda: 3)
    assert '{0:.2f}'.format(dv) == '3.00'


def test_deferred_value_unicode():
    dv = DeferredValue(lambda: 7)
    assert dv.__unicode__() == '7'
