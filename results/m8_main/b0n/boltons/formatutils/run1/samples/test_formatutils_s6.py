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


def test_construct_format_field_str_with_spec():
    assert construct_format_field_str('name', '>10', None) == '{name:>10}'


def test_construct_format_field_str_with_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'


def test_construct_format_field_str_with_conv_and_spec():
    assert construct_format_field_str('name', '.2f', 's') == '{name!s:.2f}'


def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, '>10', 'r') == ''


def test_construct_format_field_str_empty_fname():
    assert construct_format_field_str('', '', None) == '{}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

def test_split_format_str_plain_literal():
    assert split_format_str('hello') == [('hello', None)]


def test_split_format_str_single_field():
    assert split_format_str('{name}') == [('', '{name}')]


def test_split_format_str_mixed():
    assert split_format_str('Hi {name}!') == [('Hi ', '{name}'), ('!', None)]


def test_split_format_str_with_spec_and_conv():
    assert split_format_str('{x!r:>10}') == [('', '{x!r:>10}')]


def test_split_format_str_multiple_fields():
    result = split_format_str('{a}{b}')
    assert result == [('', '{a}'), ('', '{b}')]


def test_split_format_str_empty_string():
    assert split_format_str('') == []


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

def test_infer_positional_basic():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_positional_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_positional_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_with_attr():
    assert infer_positional_format_args('{.attr}') == '{0.attr}'


def test_infer_positional_with_index():
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


def test_infer_positional_multiple():
    assert infer_positional_format_args('{} {}') == '{0} {1}'


def test_infer_positional_with_mixed_named():
    assert infer_positional_format_args('{a} {} {b}') == '{a} {0} {b}'


def test_infer_positional_escaped_braces():
    assert infer_positional_format_args('{{}}') == '{{}}'


def test_infer_positional_escaped_and_arg():
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_positional_no_args():
    assert infer_positional_format_args('no args here') == 'no args here'


def test_infer_positional_numbered_untouched():
    assert infer_positional_format_args('{0} {1}') == '{0} {1}'


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

def test_get_format_args_docstring_example():
    assert get_format_args('{noun} is {1:d} years old{punct}') == (
        [(1, int)],
        [('noun', str), ('punct', str)],
    )


def test_get_format_args_empty():
    assert get_format_args('plain text') == ([], [])


def test_get_format_args_named_only():
    fargs, fkwargs = get_format_args('{a}{b}')
    assert fargs == []
    assert fkwargs == [('a', str), ('b', str)]


def test_get_format_args_positional_only():
    fargs, fkwargs = get_format_args('{0}{1}')
    assert fargs == [(0, str), (1, str)]
    assert fkwargs == []


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{a} {a} {0} {0}')
    assert fargs == [(0, str)]
    assert fkwargs == [('a', str)]


def test_get_format_args_type_mapping_float():
    fargs, _ = get_format_args('{0:f}')
    assert fargs == [(0, float)]


def test_get_format_args_type_mapping_int():
    fargs, _ = get_format_args('{0:d}')
    assert fargs == [(0, int)]


def test_get_format_args_type_mapping_str():
    fargs, _ = get_format_args('{0:s}')
    assert fargs == [(0, str)]


def test_get_format_args_unknown_type():
    fargs, _ = get_format_args('{0:xyz}')
    # last char is 'z', unknown -> str
    assert fargs == [(0, str)]


def test_get_format_args_nested_spec():
    fargs, fkwargs = get_format_args('{value:{width}}')
    # value is keyword, width is subfield; both str
    assert fkwargs == [('value', str), ('width', str)]
    assert fargs == []


def test_get_format_args_compound_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError):
        get_format_args('{}')


def test_get_format_args_index_compound_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

def test_tokenize_plain_literal():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_single_field():
    result = tokenize_format_str('{name}')
    assert len(result) == 1
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == 'name'


def test_tokenize_mixed():
    result = tokenize_format_str('Hi {name}!')
    assert len(result) == 3
    assert result[0] == 'Hi '
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'name'
    assert result[2] == '!'


def test_tokenize_resolve_pos_default():
    result = tokenize_format_str('{}')
    assert len(result) == 1
    assert result[0].fname == '0'


def test_tokenize_resolve_pos_false():
    result = tokenize_format_str('{}', resolve_pos=False)
    assert len(result) == 1
    assert result[0].fname == ''


def test_tokenize_multiple_fields():
    result = tokenize_format_str('{a}{b}')
    fields = [r for r in result if isinstance(r, BaseFormatField)]
    assert [f.fname for f in fields] == ['a', 'b']


def test_tokenize_empty_string():
    assert tokenize_format_str('') == []


# ---------------------------------------------------------------------------
# BaseFormatField
# ---------------------------------------------------------------------------

def test_baseformatfield_defaults():
    field = BaseFormatField('name')
    assert field.fname == 'name'
    assert field.base_name == 'name'
    assert field.fspec == ''
    assert field.conv is None
    assert field.subpath == []


def test_baseformatfield_positional_digit():
    field = BaseFormatField('0')
    assert field.is_positional is True


def test_baseformatfield_positional_empty():
    field = BaseFormatField('')
    assert field.is_positional is True


def test_baseformatfield_non_positional():
    field = BaseFormatField('name')
    assert field.is_positional is False


def test_baseformatfield_subpath_attr():
    field = BaseFormatField('a.b')
    assert field.base_name == 'a'
    assert field.subpath == ['b']


def test_baseformatfield_subpath_index():
    field = BaseFormatField('a[0]')
    assert field.base_name == 'a'
    assert field.subpath == ['0]']


def test_baseformatfield_fspec_and_type():
    field = BaseFormatField('x', '>10d')
    assert field.fspec == '>10d'
    assert field.type_char == 'd'
    assert field.type_func is int


def test_baseformatfield_fspec_float_type():
    field = BaseFormatField('x', '.2f')
    assert field.type_char == 'f'
    assert field.type_func is float


def test_baseformatfield_fspec_str_type():
    field = BaseFormatField('x', 's')
    assert field.type_func is str


def test_baseformatfield_fspec_unknown_type():
    field = BaseFormatField('x', 'q')
    assert field.type_func is str


def test_baseformatfield_subfields():
    field = BaseFormatField('x', '.{precision}f')
    assert field.subfields == ['precision']


def test_baseformatfield_subfields_multiple():
    field = BaseFormatField('x', '{a}.{b}')
    assert field.subfields == ['a', 'b']


def test_baseformatfield_no_subfields():
    field = BaseFormatField('x', '>10')
    assert field.subfields == []


def test_baseformatfield_conv():
    field = BaseFormatField('x', conv='r')
    assert field.conv == 'r'
    assert field.conv_func is None


def test_baseformatfield_fstr_no_spec_no_conv():
    field = BaseFormatField('name')
    assert field.fstr == '{name}'


def test_baseformatfield_fstr_with_spec():
    field = BaseFormatField('name', '>10')
    assert field.fstr == '{name:>10}'


def test_baseformatfield_fstr_with_conv():
    field = BaseFormatField('name', conv='r')
    assert field.fstr == '{name!r}'


def test_baseformatfield_str():
    field = BaseFormatField('x', '.2f', 'r')
    assert str(field) == '{x!r:.2f}'


def test_baseformatfield_repr_full():
    field = BaseFormatField('x', '.2f', 'r')
    assert repr(field) == "BaseFormatField('x', '.2f', 'r')"


def test_baseformatfield_repr_spec_only():
    field = BaseFormatField('x', '.2f')
    assert repr(field) == "BaseFormatField('x', '.2f')"


def test_baseformatfield_repr_name_only():
    field = BaseFormatField('x')
    assert repr(field) == "BaseFormatField('x')"


def test_baseformatfield_set_fname():
    field = BaseFormatField('a')
    field.set_fname('b.c')
    assert field.base_name == 'b'
    assert field.subpath == ['c']
    assert field.fname == 'b.c'


def test_baseformatfield_set_fspec():
    field = BaseFormatField('a')
    field.set_fspec('>5d')
    assert field.fspec == '>5d'
    assert field.type_char == 'd'
    assert field.type_func is int


def test_baseformatfield_set_conv():
    field = BaseFormatField('a')
    field.set_conv('s')
    assert field.conv == 's'


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

def test_deferred_value_caches_by_default():
    calls = []

    def func():
        calls.append(1)
        return 42

    dv = DeferredValue(func)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
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


def test_deferred_value_lazy():
    calls = []

    def func():
        calls.append(1)
        return 'x'

    dv = DeferredValue(func)
    assert calls == []
    dv.get_value()
    assert len(calls) == 1


def test_deferred_value_int():
    dv = DeferredValue(lambda: '123')
    assert int(dv) == 123


def test_deferred_value_float():
    dv = DeferredValue(lambda: '1.5')
    assert float(dv) == 1.5


def test_deferred_value_str():
    dv = DeferredValue(lambda: 5)
    assert str(dv) == '5'


def test_deferred_value_repr():
    dv = DeferredValue(lambda: [1, 2])
    assert repr(dv) == '[1, 2]'


def test_deferred_value_format():
    dv = DeferredValue(lambda: 42)
    assert '{:d}'.format(dv) == '42'


def test_deferred_value_format_int_conversion():
    # value is a string, format asks for integer conversion
    dv = DeferredValue(lambda: '3')
    assert '{:d}'.format(dv) == '3'


def test_deferred_value_format_float_conversion():
    dv = DeferredValue(lambda: '2')
    assert '{:f}'.format(dv) == '2.000000'


def test_deferred_value_with_str_format():
    dv = DeferredValue(lambda: 'hello')
    assert 'value: {0}'.format(dv) == 'value: hello'


def test_deferred_value_uses_cached_in_format():
    calls = []

    def func():
        calls.append(1)
        return 10

    dv = DeferredValue(func)
    format(dv, 'd')
    format(dv, 'd')
    assert len(calls) == 1
