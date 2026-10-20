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

def test_construct_format_field_str_none_fname_returns_empty():
    assert construct_format_field_str(None, 'x', 'r') == ''


def test_construct_format_field_str_basic():
    assert construct_format_field_str('name', '', None) == '{name}'


def test_construct_format_field_str_with_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'


def test_construct_format_field_str_with_spec():
    assert construct_format_field_str('num', '05d', None) == '{num:05d}'


def test_construct_format_field_str_with_conv_and_spec():
    assert construct_format_field_str('num', '.2f', 'r') == '{num!.2f}'[0:-1] + '}'
    # the exact form is '{num!r:.2f}'
    assert construct_format_field_str('num', '.2f', 'r') == '{num!r:.2f}'


def test_construct_format_field_str_empty_strings():
    # empty conv and empty spec are falsy, so no '!' or ':' added
    assert construct_format_field_str('x', '', '') == '{x}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

def test_split_format_str_plain_literal():
    assert split_format_str('hello') == [('hello', None)]


def test_split_format_str_simple_field():
    assert split_format_str('a{b}c') == [('a', '{b}'), ('c', None)]


def test_split_format_str_with_spec_and_conv():
    result = split_format_str('{x!r:>10} rest')
    assert result == [('', '{x!r:>10}'), (' rest', None)]


def test_split_format_str_multiple_fields():
    assert split_format_str('{a}{b}') == [('', '{a}'), ('', '{b}')]


def test_split_format_str_escaped_braces():
    # escaped braces produce literal text, no field
    assert split_format_str('{{literal}}') == [('{literal}', None)]


def test_split_format_str_empty():
    assert split_format_str('') == []


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

def test_infer_positional_format_args_empty():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_positional_format_args_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_positional_format_args_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_positional_format_args_multiple():
    assert infer_positional_format_args('{} and {}') == '{0} and {1}'


def test_infer_positional_format_args_mixed():
    result = infer_positional_format_args('x={} y={:d} z={}'.format('{}', '{}', '{}'))
    # Simpler direct check
    assert infer_positional_format_args('x={} y={:d} z={}') == 'x={0} y={1:d} z={2}'


def test_infer_positional_format_args_escaped_braces_preserved():
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_positional_format_args_named_untouched():
    assert infer_positional_format_args('{name} {}') == '{name} {0}'


def test_infer_positional_format_args_no_fields():
    assert infer_positional_format_args('plain text') == 'plain text'


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

def test_get_format_args_docstring_example():
    assert get_format_args("{noun} is {1:d} years old{punct}") == (
        [(1, int)],
        [('noun', str), ('punct', str)],
    )


def test_get_format_args_empty():
    assert get_format_args('plain') == ([], [])


def test_get_format_args_positional_only():
    assert get_format_args('{0} {1}') == ([(0, str), (1, str)], [])


def test_get_format_args_named_only():
    assert get_format_args('{a} {b}') == ([], [('a', str), ('b', str)])


def test_get_format_args_type_mapping_float():
    fargs, fkwargs = get_format_args('{0:f}')
    assert fargs == [(0, float)]
    assert fkwargs == []


def test_get_format_args_type_mapping_int_variants():
    for ch in 'bcdoxXn':
        fargs, fkwargs = get_format_args('{0:' + ch + '}')
        assert fargs == [(0, int)], "failed for type char %r" % ch


def test_get_format_args_type_mapping_float_variants():
    for ch in 'eEfFgGn%':
        fargs, fkwargs = get_format_args('{0:' + ch + '}')
        assert fargs == [(0, float)], "failed for type char %r" % ch


def test_get_format_args_type_mapping_str():
    fargs, _ = get_format_args('{0:s}')
    assert fargs == [(0, str)]


def test_get_format_args_unknown_type_defaults_str():
    fargs, _ = get_format_args('{0:z}')
    assert fargs == [(0, str)]


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{0} {0} {a} {a}')
    assert fargs == [(0, str)]
    assert fkwargs == [('a', str)]


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError):
        get_format_args('{}')


def test_get_format_args_anonymous_positional_raises():
    with pytest.raises(ValueError):
        get_format_args('{0} {}')


def test_get_format_args_compound_attribute_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


def test_get_format_args_subfield_named():
    # subfield in spec is added as a keyword arg with default str type
    fargs, fkwargs = get_format_args('{0:{width}}')
    assert fargs == [(0, str)]
    assert ('width', str) in fkwargs


def test_get_format_args_subfield_positional():
    fargs, fkwargs = get_format_args('{0:{1}}')
    assert (0, str) in fargs
    assert (1, str) in fargs


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

def test_tokenize_format_str_plain_literal():
    assert tokenize_format_str('plain') == ['plain']


def test_tokenize_format_str_empty():
    assert tokenize_format_str('') == []


def test_tokenize_format_str_simple():
    result = tokenize_format_str('a{b}c')
    assert result[0] == 'a'
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == 'b'
    assert result[2] == 'c'


def test_tokenize_format_str_resolve_pos():
    result = tokenize_format_str('{}')
    assert len(result) == 1
    field = result[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == '0'
    assert field.is_positional


def test_tokenize_format_str_no_resolve_pos():
    # with resolve_pos=False, an anonymous field is left with empty name
    result = tokenize_format_str('{}', resolve_pos=False)
    assert len(result) == 1
    field = result[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == ''
    assert field.is_positional


def test_tokenize_format_str_field_with_spec_and_conv():
    result = tokenize_format_str('{x!r:>10}')
    field = result[0]
    assert isinstance(field, BaseFormatField)
    assert field.fname == 'x'
    assert field.fspec == '>10'
    assert field.conv == 'r'


def test_tokenize_format_str_escaped_braces():
    assert tokenize_format_str('{{}}') == ['{}']


def test_tokenize_format_str_multiple_fields():
    result = tokenize_format_str('{a}-{b}')
    assert len(result) == 3
    assert isinstance(result[0], BaseFormatField) and result[0].fname == 'a'
    assert result[1] == '-'
    assert isinstance(result[2], BaseFormatField) and result[2].fname == 'b'


# ---------------------------------------------------------------------------
# BaseFormatField
# ---------------------------------------------------------------------------

def test_base_format_field_simple_name():
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
    assert f.base_name == ''
    assert f.is_positional is True


def test_base_format_field_attribute_path():
    f = BaseFormatField('a.b.c')
    assert f.base_name == 'a'
    assert f.subpath == ['b', 'c']
    assert f.fname == 'a.b.c'
    assert f.is_positional is False


def test_base_format_field_index_path():
    f = BaseFormatField('a[0]')
    assert f.base_name == 'a'
    assert f.subpath == ['0']


def test_base_format_field_fspec_and_type():
    f = BaseFormatField('x', '05d')
    assert f.fspec == '05d'
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_fspec_float():
    f = BaseFormatField('x', '.2f')
    assert f.type_char == 'f'
    assert f.type_func is float


def test_base_format_field_fspec_str_default():
    f = BaseFormatField('x', '')
    assert f.type_char == ''
    assert f.type_func is str


def test_base_format_field_subfields():
    f = BaseFormatField('x', '{width}')
    assert f.subfields == ['width']


def test_base_format_field_conv():
    f = BaseFormatField('x', '', 'r')
    assert f.conv == 'r'
    assert f.conv_func is None


def test_base_format_field_fstr_property():
    f = BaseFormatField('x', '.2f', 'r')
    assert f.fstr == '{x!r:.2f}'


def test_base_format_field_str():
    f = BaseFormatField('x', '>10')
    assert str(f) == '{x:>10}'


def test_base_format_field_repr_no_extra():
    f = BaseFormatField('x')
    assert repr(f) == "BaseFormatField('x')"


def test_base_format_field_repr_with_spec():
    f = BaseFormatField('x', '>10')
    assert repr(f) == "BaseFormatField('x', '>10')"


def test_base_format_field_repr_with_conv():
    f = BaseFormatField('x', '.2f', 'r')
    assert repr(f) == "BaseFormatField('x', '.2f', 'r')"


def test_base_format_field_set_fname_updates():
    f = BaseFormatField('a')
    f.set_fname('b.c')
    assert f.fname == 'b.c'
    assert f.base_name == 'b'
    assert f.subpath == ['c']
    assert f.is_positional is False


def test_base_format_field_set_fspec_updates():
    f = BaseFormatField('x')
    f.set_fspec('03d')
    assert f.fspec == '03d'
    assert f.type_char == 'd'
    assert f.type_func is int


def test_base_format_field_set_fspec_none():
    f = BaseFormatField('x', 'd')
    f.set_fspec(None)
    assert f.fspec == ''
    assert f.type_char == ''


def test_base_format_field_set_conv_updates():
    f = BaseFormatField('x')
    f.set_conv('s')
    assert f.conv == 's'
    assert f.conv_func is None


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

def test_deferred_value_basic():
    calls = []

    def compute():
        calls.append(1)
        return 42

    dv = DeferredValue(compute)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
    assert calls == [1]


def test_deferred_value_no_cache():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(compute, cache_value=False)
    assert dv.get_value() == 1
    assert dv.get_value() == 2
    assert calls == [1, 1]


def test_deferred_value_str():
    dv = DeferredValue(lambda: 123)
    assert str(dv) == '123'


def test_deferred_value_int():
    dv = DeferredValue(lambda: '17')
    assert int(dv) == 17


def test_deferred_value_float():
    dv = DeferredValue(lambda: '3.5')
    assert float(dv) == 3.5


def test_deferred_value_repr():
    dv = DeferredValue(lambda: 'abc')
    assert repr(dv) == "'abc'"


def test_deferred_value_format():
    dv = DeferredValue(lambda: 255)
    assert '{:x}'.format(dv) == 'ff'


def test_deferred_value_format_int_str():
    dv = DeferredValue(lambda: '42')
    assert '{:>5}'.format(dv) == '   42'


def test_deferred_value_unicode():
    dv = DeferredValue(lambda: 'hello')
    assert dv.__unicode__() == 'hello'


def test_deferred_value_cached_property():
    dv = DeferredValue(lambda: 5)
    assert dv._value is not 5  # initially unset sentinel
    dv.get_value()
    assert dv._value == 5


def test_deferred_value_format_fallback():
    # A value that initially fails __format__ with a spec but converts nicely.
    # Integers with unknown spec raise ValueError; DeferredValue fallback
    # converts via type_conv (str) and formats.
    dv = DeferredValue(lambda: [1, 2, 3])
    result = '{:>10}'.format(dv)
    assert result == '   [1, 2, 3]'
