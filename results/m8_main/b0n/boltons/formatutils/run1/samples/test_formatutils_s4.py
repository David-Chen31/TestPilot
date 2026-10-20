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


# ---------- construct_format_field_str ----------

def test_construct_format_field_str_basic():
    assert construct_format_field_str('name', '', None) == '{name}'


def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, 'd', 'r') == ''


def test_construct_format_field_str_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'


def test_construct_format_field_str_spec():
    assert construct_format_field_str('name', '>10', None) == '{name:>10}'


def test_construct_format_field_str_spec_and_conv():
    assert construct_format_field_str('val', '.2f', 's') == '{val!s:.2f}'


# ---------- split_format_str ----------

def test_split_format_str_plain_literal():
    assert split_format_str('hello') == [('hello', None)]


def test_split_format_str_simple_field():
    assert split_format_str('{x}') == [('', '{x}')]


def test_split_format_str_mixed():
    assert split_format_str('a{x}b{y}c') == [
        ('a', '{x}'),
        ('b', '{y}'),
        ('c', None),
    ]


def test_split_format_str_with_spec_and_conv():
    assert split_format_str('{x!r:>5}') == [('', '{x!r:>5}')]


def test_split_format_str_empty():
    assert split_format_str('') == [('', None)]


# ---------- infer_positional_format_args ----------

def test_infer_positional_basic():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_positional_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_positional_multiple():
    assert infer_positional_format_args('{} {:d}') == '{0} {1:d}'


def test_infer_positional_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_with_attr():
    assert infer_positional_format_args('{.attr}') == '{0.attr}'


def test_infer_positional_with_index():
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


def test_infer_positional_escaped_braces():
    assert infer_positional_format_args('{{}}') == '{{}}'


def test_infer_positional_mixed_escaped_and_anon():
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_positional_explicit_untouched():
    assert infer_positional_format_args('{0} {1:d}') == '{0} {1:d}'


# ---------- get_format_args ----------

def test_get_format_args_docstring_example():
    assert get_format_args("{noun} is {1:d} years old{punct}") == (
        [(1, int)],
        [('noun', str), ('punct', str)],
    )


def test_get_format_args_positional_and_named():
    assert get_format_args('{0} {name}') == ([(0, str)], [('name', str)])


def test_get_format_args_int_types():
    fargs, fkwargs = get_format_args('{0:d}{1:x}{2:b}')
    assert fargs == [(0, int), (1, int), (2, int)]
    assert fkwargs == []


def test_get_format_args_float_types():
    fargs, _ = get_format_args('{0:f}{1:e}{2:g}')
    assert fargs == [(0, float), (1, float), (2, float)]


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{x}{x}{0}{0}')
    assert fargs == [(0, str)]
    assert fkwargs == [('x', str)]


def test_get_format_args_nested_spec():
    fargs, fkwargs = get_format_args('{0:{width}}')
    assert fargs == [(0, str)]
    assert fkwargs == [('width', str)]


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError, match='anonymous positional'):
        get_format_args('{}')


def test_get_format_args_compound_raises():
    with pytest.raises(ValueError, match='compound format arg'):
        get_format_args('{a.b}')


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError, match='compound format arg'):
        get_format_args('{a[0]}')


def test_get_format_args_empty():
    assert get_format_args('no format here') == ([], [])


# ---------- tokenize_format_str ----------

def test_tokenize_format_str_basic():
    res = tokenize_format_str('a{x}b')
    assert res[0] == 'a'
    assert isinstance(res[1], BaseFormatField)
    assert res[1].fname == 'x'
    assert res[2] == 'b'


def test_tokenize_format_str_resolves_anonymous():
    res = tokenize_format_str('{}')
    assert len(res) == 1
    assert isinstance(res[0], BaseFormatField)
    assert res[0].fname == '0'


def test_tokenize_format_str_no_resolve():
    res = tokenize_format_str('{}', resolve_pos=False)
    assert len(res) == 1
    field = res[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == ''


def test_tokenize_format_str_preserves_spec_and_conv():
    res = tokenize_format_str('{x!r:>5}')
    field = res[0]
    assert field.fspec == '>5'
    assert field.conv == 'r'


def test_tokenize_format_str_empty():
    assert tokenize_format_str('') == []


def test_tokenize_format_str_only_literal():
    assert tokenize_format_str('hello') == ['hello']


# ---------- BaseFormatField ----------

def test_base_format_field_defaults():
    f = BaseFormatField('name')
    assert f.fname == 'name'
    assert f.base_name == 'name'
    assert f.subpath == []
    assert f.fspec == ''
    assert f.conv is None
    assert f.is_positional is False
    assert f.type_char == ''
    assert f.type_func is str
    assert f.subfields == []
    assert f.conv_func is None


def test_base_format_field_positional_digit():
    f = BaseFormatField('0')
    assert f.is_positional is True
    assert f.base_name == '0'


def test_base_format_field_anonymous_is_positional():
    f = BaseFormatField('')
    assert f.is_positional is True
    assert f.base_name == ''


def test_base_format_field_subpath():
    f = BaseFormatField('a.b')
    assert f.base_name == 'a'
    assert f.subpath == ['b']


def test_base_format_field_subpath_index():
    f = BaseFormatField('a[0]')
    assert f.base_name == 'a'
    assert f.subpath == ['0']


def test_base_format_field_fspec_int_type():
    f = BaseFormatField('x', 'd')
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_fspec_float_type():
    f = BaseFormatField('x', 'f')
    assert f.type_func is float


def test_base_format_field_fspec_str_type():
    f = BaseFormatField('x', 's')
    assert f.type_func is str


def test_base_format_field_fspec_unknown_type():
    f = BaseFormatField('x', 'Z')
    assert f.type_func is str


def test_base_format_field_subfields():
    f = BaseFormatField('x', '{width}')
    assert f.subfields == ['width']


def test_base_format_field_fspec_none():
    f = BaseFormatField('x', None)
    assert f.fspec == ''


def test_base_format_field_conv():
    f = BaseFormatField('x', conv='r')
    assert f.conv == 'r'
    assert f.conv_func is None


def test_base_format_field_fstr():
    f = BaseFormatField('x', '>5', 'r')
    assert f.fstr == '{x!r:>5}'


def test_base_format_field_str_equals_fstr():
    f = BaseFormatField('x', '>5')
    assert str(f) == '{x:>5}'


def test_base_format_field_repr_minimal():
    f = BaseFormatField('x')
    assert repr(f) == "BaseFormatField('x')"


def test_base_format_field_repr_with_spec():
    f = BaseFormatField('x', '>5')
    assert repr(f) == "BaseFormatField('x', '>5')"


def test_base_format_field_repr_with_spec_and_conv():
    f = BaseFormatField('x', '>5', 'r')
    assert repr(f) == "BaseFormatField('x', '>5', 'r')"


def test_base_format_field_set_fname():
    f = BaseFormatField('x')
    f.set_fname('a.b')
    assert f.base_name == 'a'
    assert f.subpath == ['b']


def test_base_format_field_set_fspec():
    f = BaseFormatField('x')
    f.set_fspec('d')
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_set_conv():
    f = BaseFormatField('x')
    f.set_conv('s')
    assert f.conv == 's'


# ---------- DeferredValue ----------

def test_deferred_value_get_value_caches():
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


def test_deferred_value_str():
    dv = DeferredValue(lambda: 42)
    assert str(dv) == '42'


def test_deferred_value_int():
    dv = DeferredValue(lambda: '42')
    assert int(dv) == 42


def test_deferred_value_float():
    dv = DeferredValue(lambda: '3.5')
    assert float(dv) == 3.5


def test_deferred_value_repr():
    dv = DeferredValue(lambda: [1, 2])
    assert repr(dv) == '[1, 2]'


def test_deferred_value_format():
    dv = DeferredValue(lambda: 3.14159)
    assert format(dv, '.2f') == '3.14'


def test_deferred_value_format_fallback():
    # Value returned as string; format with float spec should coerce.
    dv = DeferredValue(lambda: '3.14159')
    assert format(dv, '.2f') == '3.14'


def test_deferred_value_in_format_string():
    dv = DeferredValue(lambda: 42)
    assert 'value: {}'.format(dv) == 'value: 42'


def test_deferred_value_format_string_with_spec():
    dv = DeferredValue(lambda: 7)
    assert '{:04d}'.format(dv) == '0007'


def test_deferred_value_cache_disabled_calls_each_use():
    calls = []

    def func():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(func, cache_value=False)
    assert repr(dv) == '1'
    assert repr(dv) == '2'
    assert str(dv) == '3'
    assert len(calls) == 3
