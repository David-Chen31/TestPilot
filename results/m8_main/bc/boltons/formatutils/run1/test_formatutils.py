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

def test_construct_basic():
    assert construct_format_field_str('name', '', None) == '{name}'


def test_construct_with_spec():
    assert construct_format_field_str('age', 'd', None) == '{age:d}'


def test_construct_with_conv():
    assert construct_format_field_str('name', '', 'r') == '{name!r}'


def test_construct_with_conv_and_spec():
    assert construct_format_field_str('x', '.2f', 's') == '{x!s:.2f}'


def test_construct_none_fname():
    assert construct_format_field_str(None, 'd', 's') == ''
    assert construct_format_field_str(None, '', None) == ''


def test_construct_empty_spec_and_conv():
    assert construct_format_field_str('a', '', '') == '{a}'
    assert construct_format_field_str('a', None, None) == '{a}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

def test_split_plain_literal():
    assert split_format_str('hello') == [('hello', None)]


def test_split_single_field():
    assert split_format_str('{name}') == [('', '{name}')]


def test_split_literal_and_field():
    assert split_format_str('hi {name}!') == [('hi ', '{name}'), ('!', None)]


def test_split_multiple_fields():
    res = split_format_str('{a} and {b}')
    assert res == [('', '{a}'), (' and ', '{b}')]


def test_split_with_spec_and_conv():
    res = split_format_str('{x!r:>10}')
    assert res == [('', '{x!r:>10}')]


def test_split_empty_string():
    assert split_format_str('') == []


def test_split_escaped_braces():
    # Formatter.parse yields escaped braces as separate literal chunks.
    assert split_format_str('{{}}') == [('{', None), ('}', None)]


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

def test_infer_empty_braces():
    assert infer_positional_format_args('{}') == '{0}'


def test_infer_with_spec():
    assert infer_positional_format_args('{:d}') == '{0:d}'


def test_infer_with_conv():
    assert infer_positional_format_args('{!r}') == '{0!r}'


def test_infer_multiple():
    assert infer_positional_format_args('{} {} {}') == '{0} {1} {2}'


def test_infer_mixed_with_named():
    assert infer_positional_format_args('{name} {}') == '{name} {0}'


def test_infer_escaped_not_counted():
    assert infer_positional_format_args('{{}} {}') == '{{}} {0}'


def test_infer_with_attr_access():
    assert infer_positional_format_args('{.attr}') == '{0.attr}'


def test_infer_with_index_access():
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


def test_infer_no_changes():
    assert infer_positional_format_args('{0} {name}') == '{0} {name}'


def test_infer_plain_text():
    assert infer_positional_format_args('nothing here') == 'nothing here'


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

def test_get_format_args_docstring_example():
    fargs, fkwargs = get_format_args('{noun} is {1:d} years old{punct}')
    assert fargs == [(1, int)]
    assert fkwargs == [('noun', str), ('punct', str)]


def test_get_format_args_empty():
    assert get_format_args('') == ([], [])


def test_get_format_args_no_fields():
    assert get_format_args('plain text') == ([], [])


def test_get_format_args_multiple_positional():
    fargs, fkwargs = get_format_args('{0} {2:f}')
    assert fargs == [(0, str), (2, float)]
    assert fkwargs == []


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args('{a} {a} {a}')
    assert fargs == []
    assert fkwargs == [('a', str)]


def test_get_format_args_type_map():
    fargs, fkwargs = get_format_args('{0:x} {name:f} {1:s} {2} {name2:o}')
    assert fargs == [(0, int), (1, str), (2, str)]
    assert fkwargs == [('name', float), ('name2', int)]


def test_get_format_args_unknown_type_defaults_str():
    fargs, fkwargs = get_format_args('{name:q}')
    assert fkwargs == [('name', str)]


def test_get_format_args_compound_raises():
    with pytest.raises(ValueError):
        get_format_args('{a.b}')


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError):
        get_format_args('{a[0]}')


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError):
        get_format_args('{}')


def test_get_format_args_subfield():
    fargs, fkwargs = get_format_args('{a:{width}}')
    assert fkwargs == [('a', str), ('width', str)]


def test_get_format_args_nested_field_spec():
    # The type of the outer field derives from the last char of the entire
    # spec ('.{precision}f' -> 'f' -> float).
    fargs, fkwargs = get_format_args('{name:{width}.{precision}f}')
    assert fkwargs == [
        ('name', float),
        ('width', str),
        ('precision', str),
    ]


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

def test_tokenize_plain():
    assert tokenize_format_str('hello') == ['hello']


def test_tokenize_empty():
    assert tokenize_format_str('') == []


def test_tokenize_single_field():
    res = tokenize_format_str('{name}')
    assert len(res) == 1
    assert isinstance(res[0], BaseFormatField)
    assert res[0].fname == 'name'


def test_tokenize_alternating():
    res = tokenize_format_str('hi {name}!')
    assert len(res) == 3
    assert res[0] == 'hi '
    assert isinstance(res[1], BaseFormatField)
    assert res[1].fname == 'name'
    assert res[2] == '!'


def test_tokenize_resolve_pos():
    res = tokenize_format_str('{} {}')
    assert len(res) == 3
    assert isinstance(res[0], BaseFormatField)
    assert res[0].fname == '0'
    assert res[1] == ' '
    assert isinstance(res[2], BaseFormatField)
    assert res[2].fname == '1'


def test_tokenize_resolve_pos_false():
    res = tokenize_format_str('{}', resolve_pos=False)
    assert len(res) == 1
    assert isinstance(res[0], BaseFormatField)
    assert res[0].fname == ''


def test_tokenize_with_spec():
    res = tokenize_format_str('{x:.2f}')
    assert len(res) == 1
    field = res[0]
    assert field.fname == 'x'
    assert field.fspec == '.2f'
    assert field.type_char == 'f'
    assert field.type_func is float


def test_tokenize_with_conv():
    res = tokenize_format_str('{x!r}')
    field = res[0]
    assert field.conv == 'r'


# ---------------------------------------------------------------------------
# BaseFormatField
# ---------------------------------------------------------------------------

def test_bff_defaults():
    f = BaseFormatField('name')
    assert f.fname == 'name'
    assert f.base_name == 'name'
    assert f.fspec == ''
    assert f.conv is None
    assert f.subfields == []
    assert f.subpath == []
    assert f.is_positional is False
    assert f.type_char == ''
    assert f.type_func is str


def test_bff_positional_digit():
    f = BaseFormatField('0')
    assert f.is_positional is True


def test_bff_positional_empty():
    f = BaseFormatField('')
    assert f.is_positional is True
    assert f.base_name == ''


def test_bff_named_not_positional():
    f = BaseFormatField('abc')
    assert f.is_positional is False


def test_bff_subpath():
    f = BaseFormatField('a.b.c')
    assert f.base_name == 'a'
    assert f.subpath == ['b', 'c']


def test_bff_index_subpath():
    f = BaseFormatField('a[1]')
    assert f.base_name == 'a'
    assert f.subpath == ['1]']


def test_bff_fspec_type_char():
    f = BaseFormatField('x', 'd')
    assert f.type_char == 'd'
    assert f.type_func is int


def test_bff_fspec_float():
    f = BaseFormatField('x', '.3f')
    assert f.type_func is float


def test_bff_fspec_unknown_type():
    f = BaseFormatField('x', 'z')
    assert f.type_func is str


def test_bff_fspec_subfields():
    f = BaseFormatField('x', '.{precision}f')
    assert f.subfields == ['precision']
    assert f.fspec == '.{precision}f'


def test_bff_fspec_none():
    f = BaseFormatField('x', None)
    assert f.fspec == ''


def test_bff_conv():
    f = BaseFormatField('x', '', 'r')
    assert f.conv == 'r'
    assert f.conv_func is None


def test_bff_fstr():
    f = BaseFormatField('x', '.2f', 'r')
    assert f.fstr == '{x!r:.2f}'


def test_bff_str():
    f = BaseFormatField('x', '.2f')
    assert str(f) == '{x:.2f}'


def test_bff_repr_basic():
    assert repr(BaseFormatField('x')) == "BaseFormatField('x')"


def test_bff_repr_with_spec():
    assert repr(BaseFormatField('x', '.2f')) == "BaseFormatField('x', '.2f')"


def test_bff_repr_with_conv():
    assert (
        repr(BaseFormatField('x', '.2f', 'r'))
        == "BaseFormatField('x', '.2f', 'r')"
    )


def test_bff_repr_conv_no_spec():
    assert repr(BaseFormatField('x', '', 'r')) == "BaseFormatField('x', '', 'r')"


def test_bff_setters_update():
    f = BaseFormatField('x')
    f.set_fname('y.z')
    f.set_fspec('d')
    f.set_conv('s')
    assert f.fname == 'y.z'
    assert f.base_name == 'y'
    assert f.subpath == ['z']
    assert f.fspec == 'd'
    assert f.type_func is int
    assert f.conv == 's'


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

def test_dv_caches_by_default():
    calls = []

    def func():
        calls.append(1)
        return 42

    dv = DeferredValue(func)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
    assert len(calls) == 1


def test_dv_no_cache():
    calls = []

    def func():
        calls.append(1)
        return len(calls)

    dv = DeferredValue(func, cache_value=False)
    assert dv.get_value() == 1
    assert dv.get_value() == 2
    assert dv.get_value() == 3


def test_dv_int():
    dv = DeferredValue(lambda: '7')
    assert int(dv) == 7


def test_dv_float():
    dv = DeferredValue(lambda: '3.5')
    assert float(dv) == 3.5


def test_dv_str():
    dv = DeferredValue(lambda: 123)
    assert str(dv) == '123'


def test_dv_unicode():
    dv = DeferredValue(lambda: 99)
    assert dv.__unicode__() == '99'


def test_dv_repr():
    dv = DeferredValue(lambda: 'hello')
    assert repr(dv) == repr('hello')


def test_dv_format_falls_back():
    # A value that doesn't natively support a format spec; fallback
    # converts using type_conv then formats that result.
    dv = DeferredValue(lambda: '42')
    assert format(dv, '>5') == '42'.__format__('>5')


def test_dv_format_native():
    dv = DeferredValue(lambda: 3.14159)
    assert format(dv, '.2f') == '3.14'


def test_dv_format_in_str_format():
    dv = DeferredValue(lambda: 10)
    assert '{0}'.format(dv) == '10'


def test_dv_caches_object_identity():
    obj = object()
    dv = DeferredValue(lambda: obj)
    assert dv.get_value() is obj
    assert dv.get_value() is obj


# ---------------------------------------------------------------------------
# DeferredValue.__format__ fallback path (lines 352-354)
# ---------------------------------------------------------------------------

def test_dv_format_fallback_valueerror_uses_type_conv():
    # A str value with a numeric format spec raises ValueError natively;
    # the fallback converts via type_conv (float for 'f') and formats.
    dv = DeferredValue(lambda: '3.5')
    assert format(dv, '.1f') == format(3.5, '.1f')
    assert format(dv, '.1f') == '3.5'


def test_dv_format_fallback_typeerror_uses_type_conv():
    # A str value with the 'd' integer spec raises ValueError; the
    # fallback converts '42' to int then applies the spec.
    dv = DeferredValue(lambda: '42')
    assert format(dv, 'd') == '42'
    assert format(dv, '>5d') == '   42'


def test_dv_format_fallback_int_type_conv_for_hex():
    # 'x' maps to int, so a string value gets converted to int.
    dv = DeferredValue(lambda: '255')
    assert format(dv, 'x') == 'ff'
    assert format(dv, '#x') == '0xff'


def test_dv_format_fallback_with_padding_and_precision():
    dv = DeferredValue(lambda: '2.71828')
    assert format(dv, '.3f') == '2.718'
    assert format(dv, '10.3f') == '     2.718'


def test_dv_format_fallback_concise():
    # Confirm the fallback is actually triggered, not native formatting.
    dv = DeferredValue(lambda: 'not-a-number')
    with pytest.raises(ValueError):
        # int('not-a-number') raises ValueError inside the fallback too.
        format(dv, 'd')


def test_dv_format_no_spec_uses_str_type_conv():
    dv = DeferredValue(lambda: 12345)
    assert format(dv, '') == '12345'
