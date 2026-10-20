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


# ---------------------------------------------------------------- construct_format_field_str

def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, None, None) == ''
    assert construct_format_field_str(None, 'd', 'r') == ''


def test_construct_format_field_str_basic():
    assert construct_format_field_str('name', '', None) == '{name}'
    assert construct_format_field_str('name', None, None) == '{name}'


def test_construct_format_field_str_with_spec():
    assert construct_format_field_str('name', '>10', None) == '{name:>10}'


def test_construct_format_field_str_with_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'


def test_construct_format_field_str_full():
    assert construct_format_field_str('name', '>10', 's') == '{name!s:>10}'


# ---------------------------------------------------------------- split_format_str

def test_split_format_str_plain():
    assert split_format_str('hello') == [('hello', None)]


def test_split_format_str_empty():
    # Formatter().parse('') yields nothing, so result is an empty list.
    assert split_format_str('') == []


def test_split_format_str_field():
    assert split_format_str('a{name}b') == [('a', '{name}'), ('b', None)]


def test_split_format_str_field_with_spec_and_conv():
    result = split_format_str('{x!r:>5} tail')
    assert result == [('', '{x!r:>5}'), (' tail', None)]


def test_split_format_str_multiple_fields():
    # Ensures the loop processes every field (not break after first).
    result = split_format_str('a{x}b{y}c')
    assert result == [('a', '{x}'), ('b', '{y}'), ('c', None)]


def test_split_format_str_leading_literal_field():
    # start offset of first match is 0; literal before it is empty.
    assert split_format_str('{x}tail') == [('', '{x}'), ('tail', None)]


# ---------------------------------------------------------------- infer_positional_format_args

def test_infer_positional_empty():
    assert infer_positional_format_args('') == ''


def test_infer_positional_no_args():
    assert infer_positional_format_args('hello world') == 'hello world'


def test_infer_positional_simple():
    assert infer_positional_format_args('{}') == '{0}'
    assert infer_positional_format_args('{} {}') == '{0} {1}'


def test_infer_positional_multiple():
    assert infer_positional_format_args('{}-{:d}-{}') == '{0}-{1:d}-{2}'


def test_infer_positional_escaped_braces():
    assert infer_positional_format_args('{{}}') == '{{}}'
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_positional_mixed_explicit():
    # Explicit positional args are not counted; anonymous ones start at 0.
    assert infer_positional_format_args('{0} {}') == '{0} {0}'


def test_infer_positional_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_with_attr_or_index():
    assert infer_positional_format_args('{[0]}') == '{0[0]}'
    assert infer_positional_format_args('{.attr}') == '{0.attr}'


def test_infer_positional_leading_match():
    # Match starts at index 0: no prefix added, output starts with new field.
    assert infer_positional_format_args('{}abc') == '{0}abc'


def test_infer_positional_gap_between_matches():
    # text between first field and second must be preserved
    assert infer_positional_format_args('{} mid {}') == '{0} mid {1}'


def test_infer_positional_trailing_after_match():
    assert infer_positional_format_args('x{}y') == 'x{0}y'


# ---------------------------------------------------------------- get_format_args

def test_get_format_args_docstring_example():
    fargs, fkwargs = get_format_args('{noun} is {1:d} years old{punct}')
    assert fargs == [(1, int)]
    assert fkwargs == [('noun', str), ('punct', str)]


def test_get_format_args_empty():
    assert get_format_args('') == ([], [])
    assert get_format_args('just text') == ([], [])


def test_get_format_args_types():
    fargs, fkwargs = get_format_args('{0:f} {1:s} {2:b} {3:x}')
    assert fargs == [(0, float), (1, str), (2, int), (3, int)]
    assert fkwargs == []


def test_get_format_args_unknown_type_char():
    fargs, fkwargs = get_format_args('{name:z}')
    assert fargs == []
    assert fkwargs == [('name', str)]


def test_get_format_args_default_type_no_spec():
    # No spec -> default type char 's' -> str. Catches changes to default.
    fargs, fkwargs = get_format_args('{name}')
    assert fargs == []
    assert fkwargs == [('name', str)]


def test_get_format_args_default_type_spec_nonempty():
    # Spec that ends with a space-ish/non-type char -> default str.
    # Use a spec whose last char is not in _TYPE_MAP; verify type is str.
    fargs, fkwargs = get_format_args('{name:>10}')
    assert fargs == []
    assert fkwargs == [('name', str)]


def test_get_format_args_two_char_spec_type_char():
    # spec is '>d': last char is 'd' -> int (not last two chars)
    fargs, fkwargs = get_format_args('{0:>d}')
    assert fargs == [(0, int)]
    assert fkwargs == []


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{0} {0} {name} {name}')
    assert fargs == [(0, str)]
    assert fkwargs == [('name', str)]


def test_get_format_args_nested_spec():
    fargs, fkwargs = get_format_args('{0:{width}}')
    assert fargs == [(0, str)]
    assert fkwargs == [('width', str)]


def test_get_format_args_compound_arg_raises():
    with pytest.raises(ValueError) as excinfo:
        get_format_args('{a.b}')
    assert str(excinfo.value) == "encountered compound format arg: 'a.b'"


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError) as excinfo:
        get_format_args('{0[1]}')
    assert str(excinfo.value) == "encountered compound format arg: '0[1]'"


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError) as excinfo:
        get_format_args('{}')
    assert str(excinfo.value) == 'encountered anonymous positional argument'


# ---------------------------------------------------------------- tokenize_format_str

def test_tokenize_plain():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_empty():
    assert tokenize_format_str('') == []


def test_tokenize_field():
    result = tokenize_format_str('a{name}b')
    assert len(result) == 3
    assert result[0] == 'a'
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'name'
    assert result[2] == 'b'


def test_tokenize_resolves_positional():
    result = tokenize_format_str('{}-{}')
    assert result[0].fname == '0'
    assert result[1] == '-'
    assert result[2].fname == '1'


def test_tokenize_no_resolve_pos():
    result = tokenize_format_str('{}', resolve_pos=False)
    assert len(result) == 1
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == ''
    assert result[0].is_positional is True


def test_tokenize_with_spec_and_conv():
    result = tokenize_format_str('{x!r:>5}')
    assert len(result) == 1
    field = result[0]
    assert field.fname == 'x'
    assert field.fspec == '>5'
    assert field.conv == 'r'


def test_tokenize_multiple_fields_no_literal_between():
    # ensure loop iterates over all fields / not break at first literal None
    result = tokenize_format_str('{a}{b}{c}')
    assert len(result) == 3
    assert all(isinstance(t, BaseFormatField) for t in result)
    assert [t.fname for t in result] == ['a', 'b', 'c']


def test_tokenize_literal_and_multiple_fields():
    result = tokenize_format_str('x{a}y{b}z')
    assert result[0] == 'x'
    assert result[1].fname == 'a'
    assert result[2] == 'y'
    assert result[3].fname == 'b'
    assert result[4] == 'z'


# ---------------------------------------------------------------- BaseFormatField

def test_base_format_field_simple():
    field = BaseFormatField('name')
    assert field.fname == 'name'
    assert field.base_name == 'name'
    assert field.subpath == []
    assert field.fspec == ''
    assert field.conv is None
    assert field.is_positional is False
    assert field.type_char == ''
    assert field.type_func is str
    assert field.subfields == []


def test_base_format_field_positional_digit():
    field = BaseFormatField('0')
    assert field.base_name == '0'
    assert field.is_positional is True


def test_base_format_field_positional_empty():
    field = BaseFormatField('')
    assert field.base_name == ''
    assert field.is_positional is True


def test_base_format_field_subpath():
    field = BaseFormatField('a.b')
    assert field.base_name == 'a'
    assert field.subpath == ['b']


def test_base_format_field_subpath_index():
    field = BaseFormatField('a[0]')
    assert field.base_name == 'a'
    assert field.subpath == ['0]']


def test_base_format_field_spec_types():
    field = BaseFormatField('x', 'd')
    assert field.type_char == 'd'
    assert field.type_func is int

    field = BaseFormatField('x', 'f')
    assert field.type_func is float

    field = BaseFormatField('x', 's')
    assert field.type_func is str

    field = BaseFormatField('x', 'z')
    assert field.type_func is str


def test_base_format_field_set_fspec_none():
    field = BaseFormatField('x', None)
    assert field.fspec == ''


def test_base_format_field_subfields():
    field = BaseFormatField('x', '{width}.2f')
    assert field.subfields == ['width']
    assert field.fspec == '{width}.2f'


def test_base_format_field_fstr():
    field = BaseFormatField('name', '>10', 'r')
    assert field.fstr == '{name!r:>10}'


def test_base_format_field_str_and_repr():
    field = BaseFormatField('name')
    assert str(field) == '{name}'
    assert repr(field) == "BaseFormatField('name')"


def test_base_format_field_repr_with_spec():
    field = BaseFormatField('name', '>10')
    assert repr(field) == "BaseFormatField('name', '>10')"


def test_base_format_field_repr_with_conv():
    field = BaseFormatField('name', '>10', 'r')
    assert repr(field) == "BaseFormatField('name', '>10', 'r')"


def test_base_format_field_set_fname():
    field = BaseFormatField('name')
    field.set_fname('other')
    assert field.fname == 'other'
    assert field.base_name == 'other'


def test_base_format_field_set_fname_docstring():
    field = BaseFormatField('name')
    assert field.set_fname.__doc__ == "Set the field name."


def test_base_format_field_set_fspec_docstring():
    field = BaseFormatField('name')
    assert field.set_fspec.__doc__ == "Set the field spec."


def test_base_format_field_set_conv_docstring():
    field = BaseFormatField('name')
    # The docstring's second line is indented in the source; match exactly.
    assert field.set_conv.__doc__ == (
        "There are only two built-in converters: ``s`` and ``r``. They are\n"
        "        somewhat rare and appearlike ``\"{ref!r}\"``.")


def test_base_format_field_set_fspec():
    field = BaseFormatField('name')
    field.set_fspec('d')
    assert field.type_char == 'd'
    assert field.type_func is int


def test_base_format_field_set_conv():
    field = BaseFormatField('name')
    field.set_conv('s')
    assert field.conv == 's'
    assert field.conv_func is None


# ---------------------------------------------------------------- DeferredValue

def test_deferred_value_caching():
    calls = []

    def func():
        calls.append(1)
        return 42

    dv = DeferredValue(func)
    assert len(calls) == 0
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


def test_deferred_value_no_cache_does_not_store():
    dv = DeferredValue(lambda: 5, cache_value=False)
    dv.get_value()
    # value should remain unset and recomputed each time
    assert dv.get_value() == 5


def test_deferred_value_attributes():
    func = lambda: 1  # noqa: E731
    dv = DeferredValue(func)
    assert dv.func is func
    assert dv.cache_value is True


def test_deferred_value_int():
    dv = DeferredValue(lambda: '17')
    assert int(dv) == 17


def test_deferred_value_float():
    dv = DeferredValue(lambda: '3.5')
    assert float(dv) == 3.5


def test_deferred_value_str():
    dv = DeferredValue(lambda: 42)
    assert str(dv) == '42'


def test_deferred_value_repr():
    dv = DeferredValue(lambda: 'abc')
    assert repr(dv) == "'abc'"


def test_deferred_value_unicode_method():
    dv = DeferredValue(lambda: 'abc')
    assert dv.__unicode__() == 'abc'


def test_deferred_value_format_basic():
    dv = DeferredValue(lambda: 42)
    assert '{:d}'.format(dv) == '42'


def test_deferred_value_format_str_conv():
    dv = DeferredValue(lambda: 'hello')
    assert '{:>10}'.format(dv) == 'hello'.rjust(10)


def test_deferred_value_format_in_string():
    dv = DeferredValue(lambda: len([1, 2, 3]))
    output = 'there are {} items'.format(dv)
    assert output == 'there are 3 items'


def test_deferred_value_format_fallback_type_conv():
    dv = DeferredValue(lambda: '42')
    # str supports __format__ but not with 'd' -> raises, fallback to int
    assert format(dv, 'd') == '42'
