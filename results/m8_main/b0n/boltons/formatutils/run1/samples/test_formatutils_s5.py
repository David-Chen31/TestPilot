import sys

import pytest

from boltons.formatutils import (
    BaseFormatField,
    DeferredValue,
    construct_format_field_str,
    get_format_args,
    infer_positional_format_args,
    split_format_str,
    tokenize_format_str,
)


# ---------------------------------------------------------------------------
# construct_format_field_str
# ---------------------------------------------------------------------------

def test_construct_none_fname_returns_empty():
    assert construct_format_field_str(None, None, None) == ''
    assert construct_format_field_str(None, 'd', 'r') == ''


def test_construct_name_only():
    assert construct_format_field_str('noun', None, None) == '{noun}'
    assert construct_format_field_str('noun', '', None) == '{noun}'


def test_construct_with_spec():
    assert construct_format_field_str('noun', 'd', None) == '{noun:d}'


def test_construct_with_conv():
    assert construct_format_field_str('noun', None, 'r') == '{noun!r}'


def test_construct_with_spec_and_conv():
    assert construct_format_field_str('noun', 'd', 'r') == '{noun!r:d}'


def test_construct_positional():
    assert construct_format_field_str('0', '05.2f', None) == '{0:05.2f}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

def test_split_simple():
    result = split_format_str('Hello {name}, you are {age:d}')
    assert result == [
        ('Hello ', '{name}'),
        (', you are ', '{age:d}'),
    ]


def test_split_no_fields():
    assert split_format_str('plain text') == [('plain text', None)]


def test_split_consecutive_fields():
    result = split_format_str('{a}{b}')
    assert result == [('', '{a}'), ('', '{b}')]


def test_split_empty_string():
    assert split_format_str('') == [('', None)]


def test_split_with_conv():
    result = split_format_str('{x!r}')
    assert result == [('', '{x!r}')]


def test_split_leading_literal_only():
    result = split_format_str('pre')
    assert result == [('pre', None)]


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

def test_infer_empty():
    assert infer_positional_format_args('') == ''


def test_infer_no_fields():
    assert infer_positional_format_args('abc') == 'abc'


def test_infer_only_escaped_braces():
    assert infer_positional_format_args('{{}}') == '{{}}'


def test_infer_anon():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_anon_multiple():
    assert infer_positional_format_args('{} and {}') == '{0} and {1}'


def test_infer_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_with_attr():
    assert infer_positional_format_args('{.attr}') == '{0.attr}'


def test_infer_with_index():
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


def test_infer_with_nested_spec():
    assert infer_positional_format_args('{:{}}') == '{0:{1}}'


def test_infer_preserve_named():
    assert infer_positional_format_args('{named}') == '{named}'


def test_infer_preserve_numbered():
    assert infer_positional_format_args('{0}') == '{0}'


def test_infer_mixed():
    result = infer_positional_format_args('a {} b {x} c {:d}')
    assert result == 'a {0} b {x} c {1:d}'


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

def test_get_format_args_empty():
    assert get_format_args('') == ([], [])


def test_get_format_args_no_fields():
    assert get_format_args('plain') == ([], [])


def test_get_format_args_named():
    args, kwargs = get_format_args('{noun} is {punct}')
    assert args == []
    assert kwargs == [('noun', str), ('punct', str)]


def test_get_format_args_mixed():
    args, kwargs = get_format_args('{noun} is {1:d} years old{punct}')
    assert args == [(1, int)]
    assert kwargs == [('noun', str), ('punct', str)]


def test_get_format_args_dedup():
    args, kwargs = get_format_args('{a}{a}')
    assert args == []
    assert kwargs == [('a', str)]


def test_get_format_args_positional_dedup():
    args, kwargs = get_format_args('{0:d}{0}')
    assert args == [(0, int)]


def test_get_format_args_int_types():
    args, kwargs = get_format_args('{0:b}{1:d}{2:o}{3:x}{4:X}{5:n}')
    assert [a[1] for a in args] == [int, int, int, int, int, int]


def test_get_format_args_float_types():
    args, kwargs = get_format_args(
        '{0:e}{1:E}{2:f}{3:F}{4:g}{5:G}{6:n}{7:%}'
    )
    assert [a[1] for a in args] == [float] * 8


def test_get_format_args_string_type():
    args, kwargs = get_format_args('{0:s}')
    assert args == [(0, str)]


def test_get_format_args_unknown_type_char_defaults_str():
    args, kwargs = get_format_args('{0:q}')
    assert args == [(0, str)]


def test_get_format_args_named_dedup_different_types_keeps_first():
    args, kwargs = get_format_args('{x:d}{x}')
    assert kwargs == [('x', int)]


def test_get_format_args_nested_subfield():
    args, kwargs = get_format_args('{0:{width}}')
    assert args == [(0, str)]
    assert kwargs == [('width', str)]


def test_get_format_args_anon_raises():
    with pytest.raises(ValueError):
        get_format_args('{}')


def test_get_format_args_anon_with_spec_raises():
    with pytest.raises(ValueError):
        get_format_args('{:d}')


def test_get_format_args_compound_attr_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

def test_tokenize_empty():
    assert tokenize_format_str('') == []


def test_tokenize_literal_only():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_simple_field():
    result = tokenize_format_str('a{noun}b')
    assert len(result) == 3
    assert result[0] == 'a'
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'noun'
    assert result[2] == 'b'


def test_tokenize_no_literals():
    result = tokenize_format_str('{a}{b}')
    assert len(result) == 2
    assert all(isinstance(t, BaseFormatField) for t in result)
    assert result[0].fname == 'a'
    assert result[1].fname == 'b'


def test_tokenize_resolves_anon_by_default():
    result = tokenize_format_str('{}')
    assert len(result) == 1
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == '0'
    assert result[0].is_positional is True


def test_tokenize_no_resolve_anon():
    result = tokenize_format_str('{}', resolve_pos=False)
    assert len(result) == 1
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == ''
    assert result[0].is_positional is True


def test_tokenize_resolve_multiple_anon():
    result = tokenize_format_str('{}{}')
    assert [t.fname for t in result] == ['0', '1']


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

def test_base_format_field_defaults():
    f = BaseFormatField('name')
    assert f.fname == 'name'
    assert f.base_name == 'name'
    assert f.subpath == []
    assert f.fspec == ''
    assert f.conv is None
    assert f.conv_func is None
    assert f.type_char == ''
    assert f.type_func is str
    assert f.subfields == []
    assert f.is_positional is False


def test_base_format_field_positional_digit():
    f = BaseFormatField('0')
    assert f.is_positional is True
    assert f.base_name == '0'


def test_base_format_field_positional_empty():
    f = BaseFormatField('')
    assert f.is_positional is True


def test_base_format_field_subpath_attr():
    f = BaseFormatField('a.b.c')
    assert f.base_name == 'a'
    assert f.subpath == ['b', 'c']


def test_base_format_field_subpath_index():
    f = BaseFormatField('a[0]')
    assert f.base_name == 'a'
    assert f.subpath == ['0']


def test_base_format_field_spec_type():
    f = BaseFormatField('x', 'd')
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_spec_float_type():
    f = BaseFormatField('x', '.2f')
    assert f.type_char == 'f'
    assert f.type_func is float


def test_base_format_field_spec_unknown_type():
    f = BaseFormatField('x', 'q')
    assert f.type_func is str


def test_base_format_field_subfields():
    f = BaseFormatField('x', '{width}')
    assert f.subfields == ['width']


def test_base_format_field_multiple_subfields():
    f = BaseFormatField('x', '{a}{b}')
    assert f.subfields == ['a', 'b']


def test_base_format_field_fstr_property():
    f = BaseFormatField('x', 'd', 'r')
    assert f.fstr == '{x!r:d}'


def test_base_format_field_fstr_no_conv():
    f = BaseFormatField('x', 'd')
    assert f.fstr == '{x:d}'


def test_base_format_field_str():
    f = BaseFormatField('x', 'd')
    assert str(f) == '{x:d}'


def test_base_format_field_repr_simple():
    f = BaseFormatField('x')
    assert repr(f) == "BaseFormatField('x')"


def test_base_format_field_repr_with_spec():
    f = BaseFormatField('x', 'd')
    assert repr(f) == "BaseFormatField('x', 'd')"


def test_base_format_field_repr_with_conv():
    f = BaseFormatField('x', 'd', 'r')
    assert repr(f) == "BaseFormatField('x', 'd', 'r')"


def test_base_format_field_repr_with_conv_no_spec():
    f = BaseFormatField('x', '', 'r')
    assert repr(f) == "BaseFormatField('x', '', 'r')"


def test_base_format_field_reset_fname():
    f = BaseFormatField('x')
    f.set_fname('0')
    assert f.is_positional is True
    f.set_fname('name')
    assert f.is_positional is False


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

def test_deferred_value_basic():
    dv = DeferredValue(lambda: 42)
    assert dv.get_value() == 42
    assert int(dv) == 42
    assert float(dv) == 42.0
    assert str(dv) == '42'
    assert repr(dv) == '42'


def test_deferred_value_caches_by_default():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(compute)
    assert dv.get_value() == 1
    assert dv.get_value() == 1
    assert dv.get_value() == 1
    assert calls == [1]


def test_deferred_value_no_cache():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(compute, cache_value=False)
    assert dv.get_value() == 1
    assert dv.get_value() == 2
    assert dv.get_value() == 3
    assert calls == [1, 1, 1]


def test_deferred_value_format():
    dv = DeferredValue(lambda: 3.14159)
    assert '{:.2f}'.format(dv) == '3.14'


def test_deferred_value_format_fallback():
    # A value whose __format__ raises ValueError for 'd' but is convertible
    # via the presentation type mapping.
    dv = DeferredValue(lambda: '7')
    assert format(dv, 'd') == '7'


def test_deferred_value_format_int_spec_on_string():
    dv = DeferredValue(lambda: '42')
    assert format(dv, 'd') == '42'


def test_deferred_value_uses_presentation_type():
    # 'x' presentation char maps to int; value converts to int.
    dv = DeferredValue(lambda: '255')
    assert format(dv, 'x') == 'ff'


def test_deferred_value_int_conversion():
    dv = DeferredValue(lambda: '10')
    assert int(dv) == 10


def test_deferred_value_float_conversion():
    dv = DeferredValue(lambda: '10.5')
    assert float(dv) == 10.5


def test_deferred_value_repr_string():
    dv = DeferredValue(lambda: 'abc')
    assert repr(dv) == "'abc'"


def test_deferred_value_single_call_caches_across_methods():
    calls = []

    def compute():
        calls.append(1)
        return 5

    dv = DeferredValue(compute)
    assert int(dv) == 5
    assert float(dv) == 5.0
    assert str(dv) == '5'
    assert dv.get_value() == 5
    assert len(calls) == 1
